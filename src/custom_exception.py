import sys
import traceback

class CustomException(Exception):

    def __init__(self, error_message, error_detail: sys = sys):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message, error_detail)

    @staticmethod
    def get_detailed_error_message(error_message, error_detail: sys = sys):
        exc_type, exc_value, exc_tb = (
            error_detail.exc_info()
            if hasattr(error_detail, "exc_info")
            else sys.exc_info()
        )
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            return f"Error in {file_name}, line {line_number}: {error_message}"
        return f"Error: {error_message}"
    
    def __str__(self):
        return self.error_message
