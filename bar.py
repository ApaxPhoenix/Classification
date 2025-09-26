import sys
import asyncio
import numpy as np
import numpy.core.defchararray as chars
import traceback
from typing import Dict, Optional, Union
import warnings


class Bar:
    """
    A nice progress bar for showing how your training is going.

    This creates those cool progress bars you see in terminals that show
    percentage complete, how fast things are going, and how much time is left.
    Works with async code and automatically cleans up after itself.

    It's pretty flexible - you can add your own metrics like loss values
    or accuracy scores that get displayed alongside the progress.
    """

    def __init__(
            self, iterations: int, title: str = "Loading", steps: int = 40
    ) -> None:
        """
        Set up a new progress bar.

        Args:
            iterations: Total number of things you need to process
            title: What to call this progress bar (like "Training" or "Loading")
            steps: How detailed the bar should be (more steps = smoother animation)
        """
        # Store how many items we need to process total
        self.iterations: int = iterations

        # What we're calling this progress bar
        self.title: str = title

        # How many characters wide the bar should be
        self.steps: int = steps

        # Place to store extra info like loss values
        self.items: Dict[str, str] = {}

    async def update(self, batch: int, time: float, final: bool = False) -> None:
        """
        Update the progress bar with where we're at now.

        This calculates how much we've done, how fast we're going,
        and estimates how much time is left. Then it draws the
        updated progress bar on screen.

        Args:
            batch: How many items we've finished so far
            time: When we started (for calculating speed)
            final: Is this the last update? (adds a newline at the end)
        """
        # Figure out how long we've been running
        elapsed: float = np.subtract(
            asyncio.get_event_loop().time(), time
        )

        # What percentage are we done?
        percentage: float = np.divide(batch, self.iterations)

        # How fast are we processing items?
        throughput: np.array = np.where(
            np.greater(elapsed, 0),
            np.floor_divide(batch, elapsed),
            0
        )

        # How much time do we probably have left?
        eta: np.array = np.where(
            np.greater(batch, 0),
            np.divide(
                np.multiply(elapsed, np.subtract(self.iterations, batch)),
                batch
            ),
            0,
        )

        # Build the actual progress bar visual
        bar: str = chars.add(
            "|",
            chars.add(
                # Fill in the completed part with # symbols
                "".join(np.repeat("#", np.ceil(np.multiply(percentage, self.steps)))),
                chars.add(
                    # Fill the rest with spaces
                    "".join(
                        np.repeat(
                            " ",
                            np.subtract(
                                self.steps,
                                np.ceil(np.multiply(percentage, self.steps))
                            ),
                        )
                    ),
                    # Add the numbers showing progress
                    f"| {batch:03d}/{self.iterations:03d}",
                ),
            ),
        )

        # Write the complete progress line to the terminal
        sys.stdout.write(
            chars.add(
                chars.add(
                    chars.add(
                        # Main progress info with percentage, time, and speed
                        f"\r{self.title}: {bar} [{np.multiply(percentage, 100):.2f}%] in {elapsed:.1f}s "
                        f"({throughput:.1f}/s, ETA: {eta:.1f}s)",

                        # Add any extra metrics if we have them
                        np.where(
                            np.greater(np.size(self.items), 0),
                            chars.add(
                                " (",
                                chars.add(
                                    # Show all the extra metrics separated by commas
                                    ", ".join(
                                        [
                                            f"{name}: {value}"
                                            for name, value in self.items.items()
                                        ]
                                    ),
                                    ")",
                                ),
                            ),
                            "",
                        ),
                    ),
                    "",
                ),
                "",
            )
        )

        # If this is the final update, add a newline so the next output doesn't overwrite us
        if final:
            sys.stdout.write("\n")

        # Make sure everything gets printed right away
        sys.stdout.flush()

    async def postfix(self, **kwargs: Union[str, int, float]) -> None:
        """
        Add extra info to show alongside the progress bar.

        This is super handy for showing things like current loss,
        accuracy, learning rate, or whatever else you want to track.

        Args:
            **kwargs: Whatever metrics you want to display
                     (like loss=0.45, accuracy=0.89, etc.)

        Example:
            await bar.postfix(loss=0.234, lr=0.001)
            # Shows: (loss: 0.234, lr: 0.001)
        """
        # Add the new metrics to our collection
        self.items.update(kwargs)

    async def __aenter__(self) -> "Bar":
        """
        Makes this work with 'async with' statements.

        This lets you do:
            async with Bar(100, "Training") as pbar:
                # your training code here
        """
        return self

    async def __aexit__(
            self,
            exc_type: Optional[type],
            exc_val: Optional[BaseException],
            exc_tb: Optional[traceback.TracebackException],
    ) -> None:
        """
        Cleans up when exiting the 'async with' block.

        If everything went well, it shows 100% completion.
        If something went wrong, it shows a warning about the error.

        Args:
            exc_type: What kind of error happened (if any)
            exc_val: The actual error object
            exc_tb: Error traceback info
        """
        # If everything finished normally
        if exc_type is None:
            # Show final completion status
            await self.update(
                self.iterations,
                asyncio.get_event_loop().time(),
                final=True
            )
        else:
            # Something went wrong - let the user know
            warnings.warn(
                f"\n{self.title} hit an error: {exc_val}"
            )