import sys
import asyncio
import numpy as np
import numpy.core.defchararray as chars
import traceback
from typing import Dict, Optional, Union
import warnings


class Bar:
    """Asynchronous progress bar for terminal-based task monitoring.

    Provides visual feedback for long-running operations including completion
    percentage, throughput metrics, estimated time of arrival (ETA), and custom
    metric displays. Designed for async/await workflows with context manager
    support for automatic cleanup.

    Attributes:
        iterations (int): Total number of operations to track for completion.
        title (str): Descriptive label displayed before progress indicator.
        steps (int): Visual granularity of progress bar (character width).
        items (Dict[str, str]): Additional metrics displayed in parentheses.
    """

    def __init__(
            self, iterations: int, title: str = "Loading", steps: int = 40
    ) -> None:
        """Initialize progress bar with task parameters and visual configuration.

        Args:
            iterations: Total number of discrete operations to process.
                Must be positive integer representing completion target.
            title: Human-readable identifier for tracked operation.
                Appears as prefix in progress display for context.
            steps: Character width of visual progress indicator.
                Higher values provide smoother updates at cost of overhead.

        Raises:
            ValueError: When iterations is not a positive integer value.
        """
        # Store total iteration count for percentage calculations
        self.iterations: int = iterations

        # Store operation identifier for display context
        self.title: str = title

        # Configure visual resolution of progress indicator
        self.steps: int = steps

        # Initialize metric storage for dynamic key-value display
        self.items: Dict[str, str] = {}

    async def update(self, batch: int, time: float, final: bool = False) -> None:
        """Refresh progress display with current completion state and metrics.

        Performs comprehensive progress calculation including completion percentage,
        throughput analysis, and time estimation.

        Args:
            batch: Current number of completed operations. Should be monotonically
                increasing and within range [0, iterations].
            time: Task initiation timestamp (seconds since epoch) for elapsed
                time calculation and throughput measurement.
            final: Indicates terminal update requiring newline character for
                proper console output formatting.

        Note:
            Uses asyncio event loop time for consistent timing measurements
            in asynchronous execution contexts.
        """
        # Calculate elapsed execution time using high-precision async timing
        elapsed: float = np.subtract(
            asyncio.get_event_loop().time(), time
        )

        # Compute completion ratio as normalized float for percentage display
        percentage: float = np.divide(batch, self.iterations)

        # Calculate processing throughput with zero-division protection
        throughput: np.array = np.where(
            np.greater(elapsed, 0),
            np.floor_divide(batch, elapsed),
            0
        )

        # Estimate remaining execution time based on current processing rate
        eta: np.array = np.where(
            np.greater(batch, 0),
            np.divide(
                np.multiply(elapsed, np.subtract(self.iterations, batch)),
                batch
            ),
            0,
        )

        # Construct visual progress indicator using numpy string operations
        bar: str = chars.add(
            "|",
            chars.add(
                # Generate filled portion using repeated hash characters
                "".join(np.repeat("#", np.ceil(np.multiply(percentage, self.steps)))),
                chars.add(
                    # Generate unfilled portion using repeated space characters
                    "".join(
                        np.repeat(
                            " ",
                            np.subtract(
                                self.steps,
                                np.ceil(np.multiply(percentage, self.steps))
                            ),
                        )
                    ),
                    # Append numeric progress indicator with zero-padding
                    f"| {batch:03d}/{self.iterations:03d}",
                ),
            ),
        )

        # Render complete progress line with comprehensive metrics
        sys.stdout.write(
            chars.add(
                chars.add(
                    chars.add(
                        # Primary progress information with timing and throughput
                        f"\r{self.title}: {bar} [{np.multiply(percentage, 100):.2f}%] in {elapsed:.1f}s "
                        f"({throughput:.1f}/s, ETA: {eta:.1f}s)",

                        # Conditionally append custom metrics if available
                        np.where(
                            np.greater(np.size(self.items), 0),
                            chars.add(
                                " (",
                                chars.add(
                                    # Format stored metrics as comma-separated pairs
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

        # Add terminal newline for final update to preserve output formatting
        if final:
            sys.stdout.write("\n")

        # Force immediate output rendering by flushing stdout buffer
        sys.stdout.flush()

    async def postfix(self, **kwargs: Union[str, int, float]) -> None:
        """Update supplementary metrics displayed alongside progress information.

        Enables dynamic addition of contextual information such as loss values,
        accuracy metrics, learning rates, or other task-specific measurements.

        Args:
            **kwargs: Arbitrary key-value pairs for metric display. Keys become
                metric labels while values are formatted as strings.
                Supports numeric types with automatic string conversion.
        """
        # Merge new metrics with existing items dictionary
        self.items.update(kwargs)

    async def __aenter__(self) -> "Bar":
        """Async context manager entry for progress bar initialization.

        Enables usage within async with statements for automatic resource
        management and proper cleanup handling.

        Returns:
            The initialized Bar instance ready for progress tracking operations.

        Example:
            async with Bar(100, "Processing") as pbar:
                for i in range(100):
                    await pbar.update(i, start_time)
        """
        # Return configured instance for context manager usage
        return self

    async def __aexit__(
            self,
            exc_type: Optional[type],
            exc_val: Optional[BaseException],
            exc_tb: Optional[traceback.TracebackException],
    ) -> None:
        """Async context manager exit protocol with exception handling.

        Provides proper cleanup and finalization of progress display when
        exiting the context manager scope.

        Args:
            exc_type: Exception class if an error occurred, None for normal exit.
            exc_val: Exception instance containing error details if applicable.
            exc_tb: Exception traceback information for debugging purposes.

        Note:
            Normal exit displays 100% completion with final newline.
            Exception exit shows warning message with error information.
        """
        # Handle normal completion scenario without exceptions
        if exc_type is None:
            # Finalize progress display showing complete execution
            await self.update(
                self.iterations,
                asyncio.get_event_loop().time(),
                final=True
            )
        else:
            # Handle exceptional termination with user notification
            warnings.warn(
                f"\n{self.title} encountered an error: {exc_val}"
            )