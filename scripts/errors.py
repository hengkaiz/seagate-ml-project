# Custom Errors
class Error(Exception):
    pass

class CreatingTestWithoutScalerError(Error):
    def __init__(self):
        self.message = "Attempting to create test dataset without scaler. Load a scaler before creating test dataset."
        super().__init__(self.message)
    
class DayOutGreaterThanDayIn(Error):
    def __init__(self):
        self.message = "win_size must be greater than day_pred"
        super().__init__(self.message)