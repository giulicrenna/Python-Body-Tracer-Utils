import math

class Point2D():
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.point = (x, y)
        
    def distance(self, other: 'Point2D') -> float:
        h0: float = abs(self.x - other.x)
        h1: float = abs(self.y - other.y)
        
        pow_sums: float = math.pow(h0, 2) + math.pow(h1, 2)
        
        return math.sqrt(pow_sums)
    
    def equal(self, other: 'Point2D') -> bool:
        return True if ((self.x == other.x) and (self.y == other.y)) else False

    def __str__(self) -> str:
        return(f'({self.x}, {self.y})')
    
class Line2D():
    def __init__(self, point1: Point2D, point2: Point2D) -> None:
        self.point1 = point1
        self.point2 = point2

    def length(self) -> float:
        return self.point1.distance(self.point2)

    def vector(self) -> 'Point2D':
        return Point2D(self.point2.x - self.point1.x, self.point2.y - self.point1.y)

    def angle_with(self, other: 'Line2D') -> float:
        vector1 = self.vector()
        vector2 = other.vector()

        dot_product = vector1.x * vector2.x + vector1.y * vector2.y
        magnitude_product = vector1.distance(Point2D(0, 0)) * vector2.distance(Point2D(0, 0))

        if magnitude_product == 0:
            return 0.0  # Avoid division by zero

        cosine_theta = dot_product / magnitude_product
        angle_radians = math.acos(max(-1.0, min(1.0, cosine_theta)))

        return math.degrees(angle_radians)

    def __str__(self) -> str:
        return f'Line from {self.point1} to {self.point2}'
    
if __name__ == '__main__':
    p1 = Point2D(0, 0)
    p2 = Point2D(0, 1)
    
    p3 = Point2D(0, 0)
    p4 = Point2D(-1, -1)
    
    l1 = Line2D(p1, p2)
    l2 = Line2D(p3, p4)

    print(l1.angle_with(l2))
 