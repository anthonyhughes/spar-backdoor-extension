import logging

# Configure basic logging (e.g., to a file or console)
# logging.basicConfig(level=logging.ERROR, 
#                     format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def might_fail(a, b):
    # try:
        # result = a / b
    #     logging.info(f"Calculation successful: {result}")
    # except ZeroDivisionError as e:
    #     # Log the exception with a full stack trace
    #     logger.exception("A ZeroDivisionError occurred during calculation")
    #     # The program continues running after logging

    result = a / b
    logging.info(f"Calculation successful: {result}")

# Example usage
might_fail(10, 0)
# print("Program continues here.")
