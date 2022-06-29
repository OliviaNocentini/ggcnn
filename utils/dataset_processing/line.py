import numpy as np
def _gr_text_l(l, offset=(0, 0)):
    """
    Transform a single point from a Cornell file line to a pair of ints.
    :param l: Line from Cornell grasp file (str)
    :param offset: Offset to apply to point positions
    :return: Point [y, x]
    """
    x, y = l.split()
    return [int(round(float(y))) - offset[0], int(round(float(x))) - offset[1]]

class Line:
    def __init__(self, x0, y0, x1, y1):
        
        self.points = np.array([[y0,x0],[y1,x1]])
        #print("m",self.m)
        #print("q",self.q)
        
    def eval(self, x):
        m = (self.points[1,0] - self.points[0,0]) / (self.points[1,1] - self.points[0,1])
        q = self.points[0,0]
        return m*x + q
        


    def plot(self, ax, color=None):
        """
        Plot grasping rectangle.
        :param ax: Existing matplotlib axis
        :param color: matplotlib color code (optional)
        """
        
        ax.plot(self.points[:,1],self.points[:,0], color=color)

    def zoom(self, factor, center):
        """
        Zoom grasp rectangle by given factor.
        :param factor: Zoom factor
        :param center: Zoom zenter (focus point, e.g. image center)
        """
        T = np.array(
            [
                [1/factor, 0],
                [0, 1/factor]
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(T, (self.points - c).T)).T + c).astype(np.int)

    def offset(self, offset):
        """
        Offset grasp rectangle
        :param offset: array [y, x] distance to offset
        """
        self.points += np.array(offset).reshape((1, 2))

    def rotate(self, angle, center):
        """
        Rotate grasp rectangle
        :param angle: Angle to rotate (in radians)
        :param center: Point to rotate around (e.g. image center)
        """
        R = np.array(
            [
                [np.cos(-angle), np.sin(-angle)],
                [-1 * np.sin(-angle), np.cos(-angle)],
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(R, (self.points - c).T)).T + c).astype(np.int)


    @classmethod
    def load_from_cornell_file(cls, fname):
        """
        Load line from a Cornell dataset grasp file.
        :param fname: Path to text file.
        :return: line
        """
        
        with open(fname) as f:
            while True:
                # Load 4 lines at a time, corners of bounding box.
                p0 = f.readline()
                if not p0:
                    break  # EOF
                p1 = f.readline()
                try:
                    gr = np.array([
                        _gr_text_l(p0),
                        _gr_text_l(p1),
                    ])

                    return Line(gr[0,1],gr[0,0],gr[1,1],gr[1,0])

                except ValueError:
                    # Some files contain weird values.
                    raise ValueError("problem in loading line")
  
