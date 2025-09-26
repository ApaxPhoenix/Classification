import sys
import asyncio
import numpy as np
import numpy.core.defchararray as chars
import traceback
from typing import Dict, Optional, Union
import warnings


class Bar:
    """
    Async progress bar for terminal display during long-running operations.

    Creates visual progress indicators showing completion percentage, processing
    speed, and estimated time remaining. Supports custom metrics display and
    integrates with async/await patterns. Automatically handles cleanup when
    operations complete or encounter errors.

    The progress bar updates in-place on the terminal and can display additional
    metrics like loss values or accuracy scores alongside the main progress.
    """

    def __init__(
            self, iterations: int, title: str = "Loading", steps: int = 40
    ) -> None:
        """
        Initialize progress bar with operation parameters.

        Args:
            iterations: Total number of items to process
            title: Display label for the progress operation
            steps: Character width of progress bar (higher = smoother updates)
        """
        # Total number of operations to complete
        self.iterations: int = iterations

        # Display label for this progress bar
        self.title: str = title

        # Visual width of progress bar in characters
        self.steps: int = steps

        # Storage for additional metrics to display
        self.items: Dict[str, str] = {}

    async def update(self, batch: int, time: float, final: bool = False) -> None:
        """
        Refresh progress display with current completion status.

        Calculates completion percentage, processing throughput, and estimated
        time remaining. Renders updated progress bar to terminal with all
        current metrics.

        Args:
            batch: Number of items completed so far
            time: Operation start timestamp for speed calculation
            final: Whether this is the final update (adds newline)
        """
        # Calculate elapsed time since start
        elapsed: float = np.subtract(
            asyncio.get_event_loop().time(), time
        )

        # Calculate completion percentage
        percentage: float = np.divide(batch, self.iterations)

        # Calculate processing rate in items per second
        throughput: np.array = np.where(
            np.greater(elapsed, 0),
            np.floor_divide(batch, elapsed),
            0
        )

        # Estimate remaining time based on current rate
        eta: np.array = np.where(
            np.greater(batch, 0),
            np.divide(
                np.multiply(elapsed, np.subtract(self.iterations, batch)),
                batch
            ),
            0,
        )

        # Construct visual progress bar representation
        bar: str = chars.add(
            "|",
            chars.add(
                # Filled portion using hash characters
                "".join(np.repeat("#", np.ceil(np.multiply(percentage, self.steps)))),
                chars.add(
                    # Empty portion using spaces
                    "".join(
                        np.repeat(
                            " ",
                            np.subtract(
                                self.steps,
                                np.ceil(np.multiply(percentage, self.steps))
                            ),
                        )
                    ),
                    # Progress counter display
                    f"| {batch:03d}/{self.iterations:03d}",
                ),
            ),
        )

        # Output complete progress line to terminal
        sys.stdout.write(
            chars.add(
                chars.add(
                    chars.add(
                        # Core progress information with timing
                        f"\r{self.title}: {bar} [{np.multiply(percentage, 100):.2f}%] in {elapsed:.1f}s "
                        f"({throughput:.1f}/s, ETA: {eta:.1f}s)",

                        # Additional metrics if available
                        np.where(
                            np.greater(np.size(self.items), 0),
                            chars.add(
                                " (",
                                chars.add(
                                    # Format custom metrics as comma-separated list
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

        # Add newline for final update to prevent overwriting
        if final:
            sys.stdout.write("\n")

        # Force immediate output to terminal
        sys.stdout.flush()

    async def postfix(self, **kwargs: Union[str, int, float]) -> None:
        """
        Update additional metrics displayed with progress bar.

        Accepts arbitrary key-value pairs for displaying supplementary
        information like training loss, accuracy, or other relevant metrics
        alongside the main progress indicator.

        Args:
            **kwargs: Metric names and values to display

        Example:
            await bar.postfix(loss=0.234, accuracy=0.891)
        """
        # Update metrics dictionary with new values
        self.items.update(kwargs)

    async def __aenter__(self) -> "Bar":
        """
        Enable usage as async context manager.

        Returns:
            Bar instance for use within async context block

        Example:
            async with Bar(total_items, "Processing") as progress:
                # processing logic here
        """
        return self

    async def __aexit__(
            self,
            exc_type: Optional[type],
            exc_val: Optional[BaseException],
            exc_tb: Optional[traceback.TracebackException],
    ) -> None:
        """
        Handle cleanup when exiting async context manager.

        On successful completion, displays final progress state. On exception,
        shows warning message about the encountered error.

        Args:
            exc_type: Exception class if error occurred
            exc_val: Exception instance that was raised
            exc_tb: Traceback information for the exception
        """
        # Handle normal completion
        if exc_type is None:
            # Display final completion state
            await self.update(
                self.iterations,
                asyncio.get_event_loop().time(),
                final=True
            )
        else:
            # Handle error case with warning message
            warnings.warn(
                f"\n{self.title} encountered error: {exc_val}"
            )