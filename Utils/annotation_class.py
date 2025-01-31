
class Annotation:
        def __init__(self, path_name, num_faces, boxes):
            self.path_name = path_name
            self.num_faces = num_faces
            self.boxes = boxes

        def __repr__(self):
            return f"Annotation(name='{self.path_name}', face count={self.num_faces}), boxes={self.boxes}"