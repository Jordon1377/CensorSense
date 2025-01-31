class BoundingBox:
    def __init__(self, x1, y1, w, h, invalid, blur=None, expression=None, illumination=None, occlusion=None, pose=None):
        self.x1 = x1
        self.y1 = y1
        self.w = w
        self.h = h
        self.invalid = invalid
        self.blur = blur
        self.expression = expression
        self.illumination = illumination
        self.occlusion = occlusion
        self.pose = pose

    def __repr__(self):
        return f"Box(coords='{(self.x1, self.y1, self.w, self.h)}', invalid?=({self.invalid}), b/e/i/o/p={(self.blur, self.expression, self.illumination, self.occlusion, self.pose)}"