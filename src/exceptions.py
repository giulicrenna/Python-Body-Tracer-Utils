class InvalidPathException(Exception):
    def __init__(self, *args: object) -> None:
        self.message: str = "String must to be a valid path."
        super().__init__(*args)
        
class VideoReadingException(Exception):
    def __init__(self, *args: object) -> None:
        self.message: str = "Could not read video."
        super().__init__(*args)