from numpy import array, cos, dot, identity, outer, sin, transpose, zeros

def dcm_from_axisangle(axis, angle):
    """ Returns a Direction Cosine Matrix (DCM) corresponding to a specific principal axis and angle of rotation.
    
    Inputs:
    Unit vector axis of rotation, axis: numpy array
    Angle of rotation, angle: float
    
    Outputs:
    Direction cosine matrix, numpy array
    
    """
    return cos(angle) * identity(3) + (1 - cos(angle)) * outer(axis, axis) - sin(angle) * ss_cross(axis)
    

def dcm_from_euler(rot_order):
    """ Forms Direction Cosine Matrix (DCM) and derivative functions from a sequence of Euler angle rotations.
    
    Inputs:
    1-indexed order of Euler angle rotations, rot_order: numpy array
    
    For a "1-2-3" rotation, the input to this function should be specified as array([1, 2, 3])
    
    Outputs:
    Direction cosine matrix function, dcm: numpy array (3, ) -> numpy array (3, 3)
    Direction cosine matrix gradient function, graddcm: numpy array (3, ) -> numpy array (3, 3, 3)
    Direction cosine matrix hessian function, hessdcm: numpy array (3, ) -> numpy array (3, 3, 3, 3)
    """
    
    def dcm(xi):
        """ Direction cosine matrix
        
        Inputs: 
        Ordered Euler angles, xi: numpy array
        
        Outputs:
        Direction cosine matrix: numpy array
        """
        
        return(dot(elem_euler_rot(rot_order[2], xi[2]), dot(elem_euler_rot(rot_order[1], xi[1]), elem_euler_rot(rot_order[0], xi[0]))))

    def graddcm(xi):
        """ Direction cosine matrix gradient
        
        Inputs: 
        Ordered Euler angles, xi: numpy array
        
        Outputs:
        Direction cosine matrix gradient: numpy array
        """
        
        layer1 = dot(elem_euler_rot(rot_order[2], xi[2]), dot(elem_euler_rot(rot_order[1], xi[1]), d_elem_euler_rot(rot_order[0], xi[0])))
        layer2 = dot(elem_euler_rot(rot_order[2], xi[2]), dot(d_elem_euler_rot(rot_order[1], xi[1]), elem_euler_rot(rot_order[0], xi[0])))
        layer3 = dot(d_elem_euler_rot(rot_order[2], xi[2]), dot(elem_euler_rot(rot_order[1], xi[1]), elem_euler_rot(rot_order[0], xi[0])))
        
        return transpose(array([layer1, layer2, layer3]), (1, 2, 0))
    
    def hessdcm(xi):
        """ Direction cosine matrix hessian
        
        Inputs: 
        Ordered Euler angles, xi: numpy array
        
        Outputs:
        Direction cosine matrix hessian: numpy array
        """
        
        layer1 = dot(elem_euler_rot(rot_order[2], xi[2]), dot(elem_euler_rot(rot_order[1], xi[1]), dd_elem_euler_rot(rot_order[0], xi[0])))
        layer2 = dot(elem_euler_rot(rot_order[2], xi[2]), dot(d_elem_euler_rot(rot_order[1], xi[1]), d_elem_euler_rot(rot_order[0], xi[0])))
        layer3 = dot(d_elem_euler_rot(rot_order[2], xi[2]), dot(elem_euler_rot(rot_order[1], xi[1]), d_elem_euler_rot(rot_order[0], xi[0])))
        cut1 = transpose(array([layer1, layer2, layer3]), (1, 2, 0))
    
        layer1 = dot(elem_euler_rot(rot_order[2], xi[2]), dot(d_elem_euler_rot(rot_order[1], xi[1]), d_elem_euler_rot(rot_order[0], xi[0])))
        layer2 = dot(elem_euler_rot(rot_order[2], xi[2]), dot(dd_elem_euler_rot(rot_order[1], xi[1]), elem_euler_rot(rot_order[0], xi[0])))
        layer3 = dot(d_elem_euler_rot(rot_order[2], xi[2]), dot(d_elem_euler_rot(rot_order[1], xi[1]), elem_euler_rot(rot_order[0], xi[0])))
        cut2 = transpose(array([layer1, layer2, layer3]), (1, 2, 0))
    
        layer1 = dot(d_elem_euler_rot(rot_order[2], xi[2]), dot(elem_euler_rot(rot_order[1], xi[1]), d_elem_euler_rot(rot_order[0], xi[0])))
        layer2 = dot(d_elem_euler_rot(rot_order[2], xi[2]), dot(d_elem_euler_rot(rot_order[1], xi[1]), elem_euler_rot(rot_order[0], xi[0])))
        layer3 = dot(dd_elem_euler_rot(rot_order[2], xi[2]), dot(elem_euler_rot(rot_order[1], xi[1]), elem_euler_rot(rot_order[0], xi[0])))
        cut3 = transpose(array([layer1, layer2, layer3]), (1, 2, 0))
    
        return transpose(array([cut1, cut2, cut3]), (1, 2, 3, 0))
    
    
    return dcm, graddcm, hessdcm

def elem_euler_rot(axis,angle):
    """ Elemenatry Euler rotation matrix

    Inputs: 
    1-indexed axis of rotation, axis: int (1, 2, 3)
    Angle of rotation, angle: float

    Outputs:
    Elementary Euler rotation: numpy array
    """
    
    rot_1 = array([[1, 0, 0], [0, cos(angle), sin(angle)], [0, -sin(angle), cos(angle)]])
    rot_2 = array([[cos(angle), 0, -sin(angle)], [0, 1, 0], [sin(angle), 0, cos(angle)]])
    rot_3 = array([[cos(angle), sin(angle), 0], [-sin(angle), cos(angle), 0], [0, 0, 1]])

    dcm_tensor = array([rot_1, rot_2, rot_3])

    return dcm_tensor[axis-1]

def d_elem_euler_rot(axis, angle):
    """ Elemenatry Euler rotation matrix gradient

    Inputs: 
    1-indexed axis of rotation, axis: int (1, 2, 3)
    Angle of rotation, angle: float

    Outputs:
    Elementary Euler rotation gradient: numpy array
    """
        
    d_rot_1 = array([[0, 0, 0], [0, -sin(angle), cos(angle)], [0, -cos(angle), -sin(angle)]])
    d_rot_2 = array([[-sin(angle), 0, -cos(angle)], [0, 0, 0], [cos(angle), 0, -sin(angle)]])
    d_rot_3 = array([[-sin(angle), cos(angle), 0], [-cos(angle), -sin(angle), 0], [0, 0, 0]])
    
    d_dcm_tensor = array([d_rot_1, d_rot_2, d_rot_3])
    return d_dcm_tensor[axis-1]

def dd_elem_euler_rot(axis, angle):
    """ Elemenatry Euler rotation matrix hessian

    Inputs: 
    1-indexed axis of rotation, axis: int (1, 2, 3)
    Angle of rotation, angle: float

    Outputs:
    Elementary Euler rotation hessian: numpy array
    """
        
    dd_rot_1 = array([[0, 0, 0], [0, -cos(angle), -sin(angle)], [0, sin(angle), -cos(angle)]])
    dd_rot_2 = array([[-cos(angle), 0, sin(angle)], [0, 0, 0], [-sin(angle), 0, -cos(angle)]])
    dd_rot_3 = array([[-cos(angle), -sin(angle), 0], [sin(angle), -cos(angle), 0], [0, 0, 0]])
    
    dd_dcm_tensor = array([dd_rot_1, dd_rot_2, dd_rot_3])
    return dd_dcm_tensor[axis-1]

def euler_to_ang(rot_order):
    """ Forms transformation from Euler angle rates to angular velocities
    
    Inputs:
    1-indexed order of Euler angle rotations, rot_order: numpy array
    
    For a "1-2-3" rotation, the input to this function should be specified as array([1, 2, 3])
    
    Outputs:
    Euler angle rates to angular velocities matrix function, T: numpy array (3, ) -> numpy array (3, 3)
    Euler angle rates to angular velocities matrix gradient function, gradT: numpy array (3, ) -> numpy array (3, 3, 3)
    """
    
    def T(xi):
        """ Euler angle rates to angular velocities transformation matrix
        
        Inputs:
        Ordered Euler angles, xi: numpy array
        
        Outputs:
        Euler angle rates to angular velocities transformation matrix: numpy array
        """
        
        col1 = dot(elem_euler_rot(rot_order[2], xi[2]), dot(elem_euler_rot(rot_order[1], xi[1]), evec(3, rot_order[0]-1)))
        col2 = dot(elem_euler_rot(rot_order[2], xi[2]), evec(3, rot_order[1]-1))
        col3 = evec(3, rot_order[2]-1)

        return array([col1, col2, col3]).T

    def gradT(xi):
        """ Euler angle rates to angular velocities transformation matrix gradient
        
        Inputs:
        Ordered Euler angles, xi: numpy array
        
        Outputs:
        Euler angle rates to angular velocities transformation matrix gradient: numpy array
        """
        
        layer1 = zeros((3,3))
        
        l2_c1 = dot(elem_euler_rot(rot_order[2], xi[2]), dot(d_elem_euler_rot(rot_order[1], xi[1]), evec(3, rot_order[0]-1)))
        l2_c2 = zeros(3)
        l2_c3 = zeros(3)
        layer2 = array([l2_c1, l2_c2, l2_c3]).T 
        
        l3_c1 = dot(d_elem_euler_rot(rot_order[2], xi[2]), dot(elem_euler_rot(rot_order[1], xi[1]), evec(3, rot_order[0]-1)))
        l3_c2 = dot(d_elem_euler_rot(rot_order[2], xi[2]), evec(3, rot_order[1]-1))
        l3_c3 = zeros(3)
        layer3 = array([l3_c1, l3_c2, l3_c3]).T
        
        return transpose(array([layer1, layer2, layer3]), (1, 2, 0))
    
    return T, gradT

def evec(length,idx):
    """ Elementary basis vector
    
    Inputs:
    Length, length: int
    Index, idx: int

    Outputs:
    Elementary basis vector, v: numpy array
    """

    v = zeros(length)
    v[idx] = 1

    return v

def ss_cross(v):
    """ Skew-symmetric cross operator matrix 
    
    Inputs: 
    Vector in R3, v: numpy array
    
    Outputs:
    Skew-symmetric cross matrix, v_cross: numpy array
    """
    v_cross = array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    return v_cross