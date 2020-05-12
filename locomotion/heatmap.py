## Copyright Mechanisms Underlying Behavior Lab, Singapore
## https://mechunderlyingbehavior.wordpress.com/

## heatmap.py is part of the locomotion package comparing animal behaviours, developed
## to support the work discussed in the paper "Computational geometric tools for
## modeling inherent variability in animal behavior" by MT Stamps, S Go, and AS Mathuru.

## This python script contains methods for computing the Conformal Spatiotemporal
## Distance (CSD) between the heatmaps of two animals representing the amount of time
## each animal spends in a given location. Specifically, this script contains methods
## for constructing the heatmap of an animal give its raw coordinate and timestamp data,
## an approximately uniform/regular triangulation of its corresponding surface in R^3,
## a conformal "flattening" of the that surface to the unit disk.  There are also
## methods for finding the alignment between two surfaces given for a given rotation
## between their respective flattenings in the unit disk and methods for measuring the
## symmetric distortion energy of such an alignment.

from math import ceil, exp, log, sin, asin, pi, acosh, cosh, sinh, cos, acos, atanh, tanh
from numpy import min, mean, std, array, linalg, dot, cross
from scipy.optimize import minimize_scalar
import locomotion.write as write
import locomotion.animal as animal
from locomotion.animal import throwError

import igl

#Static Variables
PERTURBATION = 0.000000001
TOLERANCE = 0.00001


################################################################################  
### METHOD FOR INITIALIZING HEAT MAP AND SURFACE DATA FOR EACH ANIMAL OBJECT ###
################################################################################


def getSurfaceData(animal_obj, grid_size, start_time=None, end_time=None):
  """ Computes the heatmap for a given animal representing the amount
      of time the animal spent in each location during a specified time
      interval, an approximately regular Delaunay triangulation of
      the corresponding surface, and a conformal flattening of that
      triangulation to the unit disk.

   :Parameters:
    animal_obj : animal object 
      from animal.py, initialized
    grid_size : float or int
      specifies the bin size for calculating the heatmap
      must divide both X_DIM and Y_DIM
      smaller values yield finer triangulations and larger values yield coarser triangulations
    start/end_time : float or int
      time in minutes. If unspecified, start/end time for the experiment will be used
  """
  #Check if start_time or end_time need to be set:
  if start_time == None:
    start_time = animal_obj.getExpStartTime()
  if end_time == None:
    end_time = animal_obj.getExpEndTime()

  #store given parameters
  animal_obj.setGridSize(grid_size)
  animal_obj.setPerturbation(PERTURBATION)
  animal_obj.setTolerance(TOLERANCE)
  
  print("Calculating heatmap for %s..." % animal_obj.getName())
  
  #calculuate heatmap
  frequencies = getFrequencies(animal_obj, start_time, end_time)

  print("Calculating triangulation for %s..." % animal_obj.getName())
  
  #get and record vertices
  original_coordinates = getVertexCoordinates(animal_obj, frequencies)
  animal_obj.setNumVerts(len(original_coordinates))
  animal_obj.setRegularCoordinates(original_coordinates)
  
  #get and record triangles
  triangles = getTriangles(animal_obj)
  animal_obj.setTriangulation(triangles)

  #calculate and store colors for output file
  colors = getColors(animal_obj)
  animal_obj.setColors(colors)

  print("Calculating flattened coordinates for %s..." % animal_obj.getName())
  
  #calculate and record flattened coordinates of triangulation
  flattened_coordinates = getFlatCoordinates(animal_obj) 
  animal_obj.setFlattenedCoordinates(flattened_coordinates)

  
#######################################################################################  
### METHODS FOR CALCULATING HEAT MAPS AND THEIR CORRESPONDING TRIANGULATED SURFACES ###
#######################################################################################  


def getFrequencies(animal_obj, start_time, end_time):
  """ Gathers the frequency data for approximating the heat map representing
      the amount of time an animal spent in each location of the assay 
      chamber over a specified time interval.

   :Parameters:
    animal_obj : animal object, initialized
    start_time : float, time in minutes
    end_time : float, time in minutes

   :Returns:
     two-dimensional array of ints counting the number of frames the animal
     spent in each square chamber of the bounding rectangle during the 
     specified time interval
  """

  #set or get relevant parameters
  start_frame = animal.getFrameNum(animal_obj, start_time)
  end_frame = animal.getFrameNum(animal_obj, end_time)
  perturb = animal_obj.getPerturbation()
  grid_size = animal_obj.getGridSize()
  x_dim, y_dim = animal_obj.getDims()
  num_x_grid, num_y_grid = animal_obj.getNumGrids()
  X = animal_obj.getRawVals('X', start_frame, end_frame)
  Y = animal_obj.getRawVals('Y', start_frame, end_frame)

  #initialize frequency matrix
  freqency_matrix = [[0 for j in range(num_y_grid)] for i in range(num_x_grid)]
  
  #check that coordinate data is within the specified bounds
  x_max = max(X)
  x_offset = max(x_max - x_dim, 0) + perturb
  y_max = max(Y)
  y_offset = max(y_max - y_dim, 0) + perturb
  
  #iterate through each frame, adjust out-of-bounds data, and update frequency matrix
  for i in range(len(X)):
    x = X[i] - x_offset
    if x < 0:
      print("WARNING: X data is out of bounds. Frame #%d, x=%f" % (i+1, X[i]))
      x = 0
    x_index = int(x/grid_size)
    y = Y[i] - y_offset
    if y < 0:
      print("WARNING: Y data is out of bounds. Frame #%d, x=%f" % (i+1, Y[i]))
      y = 0
    y_index = int(y/grid_size)
    freqency_matrix[x_index][y_index] += 1

  return freqency_matrix


def getZDim(animal_obj):
  """ Calculates the vertical bound for a heatmap surface
    We set it to be the smaller of the two horizontal dimensions, but it
    can be set to specified value depending on the context.

   :Parameter:
     animal_obj : animal object, initialized

   :Returns:
     int, value of vertical dimension
  """
  
  return min(animal_obj.getDims())


def getVertexCoordinates(animal_obj, freqs):
  """ Calculates the vertex coordinates for a triangulation of the surface 
      corresponding to a heat map.

    :Parameters:
      animal_obj : animal object, initialized
      freqs : 2D array of ints
        Frequency data for heatmap

    :Returns:
      list of triples of floats, specifying the x-, y-, and z-coordinates of the vertices
      for a triangulation of the surface corresponding to a heat map
  """

  #gather relevant parameters
  grid_size = animal_obj.getGridSize()
  x_dim, y_dim = animal_obj.getDims()
  num_x_grid,num_y_grid = animal_obj.getNumGrids()

  #normalize the values to floats between 0 and a specified z-dimension
  m = mean(freqs)
  s = std(freqs)
  z_dim = getZDim(animal_obj)
  for i in range(len(freqs)):
    freqs[i] = animal.normalize(freqs[i],m,s)
    freqs[i] = list(map(lambda x : z_dim*x, freqs[i]))

  #initialize list of coordinates to return
  coordinates = []

  #append coordinates for the lower left corner of each square in the heat map grid
  for i in range(num_x_grid):
    for j in range(num_y_grid):
      coordinates.append([i*grid_size, j*grid_size, freqs[i][j]])

  return coordinates


def getBoundaryVertices(animal_obj):
  """ Returns the subset of boundary vertices from a list of vertex coordinates

    :Parameters:
      animal_obj : animal object, initialized with regular coordinates updated

    :Returns:
      list of ints specifying the indices of the boundary vertices within the 
      regular coordinates of the animal in counterclockwise order starting with
      the bottom left corner
  """
  
  #gather relevant parameters
  grid_size = animal_obj.getGridSize()
  x_dim, y_dim = animal_obj.getDims()
  coordinates = animal_obj.getRegularCoordinates()

  #initialize lists for each edge of the boundary rectangle
  lower_edge, upper_edge, left_edge, right_edge = [], [], [], []

  #iterate through list of vertex coordinates and sort boundary vertices into their respective edge lists
  for c in coordinates:
    if c[0] == 0.0:
      left_edge.append(c)
    if c[0] == x_dim-grid_size:
      right_edge.append(c)
    if c[1] == 0.0:
      lower_edge.append(c)
    if c[1] == y_dim-grid_size:
      upper_edge.append(c)

  #initialize return list
  boundary_vertices = []  

  #arrange boundary vertices in counter-clockwise order
  lower_edge.sort(key=lambda c: c[0])
  upper_edge.sort(key=lambda c: c[0])
  left_edge.sort(key=lambda c: c[1])
  right_edge.sort(key=lambda c: c[1])
  for i in range(len(lower_edge)-1):
    boundary_vertices.append(coordinates.index(lower_edge[i]))
  for i in range(len(right_edge)-1):
    boundary_vertices.append(coordinates.index(right_edge[i]))
  for i in range(len(upper_edge)-1):
    boundary_vertices.append(coordinates.index(upper_edge[-i-1]))
  for i in range(len(left_edge)-1):
    boundary_vertices.append(coordinates.index(left_edge[-i-1]))

  return boundary_vertices


def getTriangles(animal_obj):
  """ Computes a basic triangulation on the regular coordinates of an animal

    :Parameters:
      animal_obj : animal object, initialized with regular coordinates set/updated

    :Returns:
      list of triples of ints, specifying the indices of the vertices for each triangle in
      in the triangulation of a surface
  """
  #store relevant parameters
  num_x_grid,num_y_grid = animal_obj.getNumGrids()

  #initialize triangle list
  triangles = []
  
  #iterate through lower left corners of grid and append canonical triangles
  for i in range(num_x_grid-1):
    for j in range(num_y_grid-1):
      triangles.append([i*num_y_grid+j,(i+1)*num_y_grid+j,(i+1)*num_y_grid+(j+1)])
      triangles.append([i*num_y_grid+j,(i+1)*num_y_grid+(j+1),i*num_y_grid+(j+1)])

  return triangles
  

def getColors(animal_obj):
  """ Calculates color for rendering each triangle in the triangulation of an animal according 
    to the average height of the regular coordinates of its vertices 

    :Parameters:
      animal_obj : animal object, initialized with regular coordinates and triangulation set/updated

    :Returns:
      list of triples of floats, specifying the RGB coordinates for each triangle in
      in the triangulation associated to an animals heat map
  """

  #gather relevant parameters
  coordinates = animal_obj.getRegularCoordinates()
  triangles = animal_obj.getTriangulation()

  #initialize return list
  colors = []

  #extract the heights (z-coordinates) of each vertex in the triangulation
  heights = [c[2] for c in coordinates]

  #gather basic statistics
  min_height = min(heights) 
  max_height = max(heights)
  mid_height = (min_height+max_height)/2

  #assign a color to each triangle based on the average height of the regular coordinates of its vertices
  for triangle in triangles:
    color = [1.0,1.0,0]
    height = mean([heights[v] for v in triangle])
    if height > mid_height:
      color[1] -= (height-mid_height)/(max_height-mid_height)
    else:
      color[0] -= (mid_height-height)/(mid_height-min_height)
      color[1] -= (mid_height-height)/(mid_height-min_height)
      color[2] += (mid_height-height)/(mid_height-min_height)
    colors.append(color)
    
  return colors


####################################################################################  
### METHODS FOR CALCULATING CONFORMAL FLATTENINGS OF TRIANGULATIONS TO UNIT DISK ###
#################################################################################### 


def mobius(u, v, a, b):
  #this is a helper method for the getFlatCoordinates method below
  return [((u-a)*(a*u+b*v-1)+(v-b)*(a*v-b*u))/((a*u+b*v-1)**2+(a*v-b*u)**2), ((v-b)*(a*u+b*v-1)-(u-a)*(a*v-b*u))/((a*u+b*v-1)**2+(a*v-b*u)**2)]


def getFlatCoordinates(animal_obj):
  """ Calculates the vertex coordinates for the triangulation of an animal from its corresponding circle packing in the unit disk

    :Parameters:
      animal_obj : animal object, initialized with regular coordinates and triangulation set/updated

    :Returns:
      list of pairs of floats, specifying the x- and y-coordinates of the vertices of a triangulation that has been conformally flattened
      to the unit disk
  """

  #store relevant parameters
  v, f = array(animal_obj.getRegularCoordinates()), array(animal_obj.getTriangulation())
    
  # boundary_vertices = getBoundaryVertices(animal_obj)
  boundary_vertices = igl.boundary_loop(f)
  tolerance = animal_obj.getTolerance()

  # ========== START: INSERT LIBIGL CODE HERE FOR CALCULATING THE CONFORMAL FLATTENING ==============
  flattened_boundary = igl.map_vertices_to_circle(v, boundary_vertices)
  print(flattened_boundary)
  flat_coordinates = igl.harmonic_weights(v, f, boundary_vertices, flattened_boundary, 1)
  # ========== END: INSERT LIBIGL CODE HERE FOR CALCULATING THE CONFORMAL FLATTENING ================
  
  # convert return list from polar coordinates to cartesian coordinates
  flat_coordinates = [[tanh(c[0])*cos(c[1]),tanh(c[0])*sin(c[1])] for c in flat_coordinates]
  flat_coordinates = [[c[0]/(1+(1-c[0]**2-c[1]**2)**0.5),c[1]/(1+(1-c[0]**2-c[1]**2)**0.5)] for c in flat_coordinates]

  # apply a conformal automorphism (Mobius transformation) of the unit disk that moves the center of mass of the flattened coordinates to the origin
  p = mean([c[0] for c in flat_coordinates])
  q = mean([c[1] for c in flat_coordinates])
  while p**2+q**2 > tolerance:
    for i in range(len(flat_coordinates)):
      x = flat_coordinates[i][0]
      y = flat_coordinates[i][1]
      flat_coordinates[i] = mobius(x,y,p,q)
      flat_coordinates[i].append(0)
    p = mean([c[0] for c in flat_coordinates])
    q = mean([c[1] for c in flat_coordinates])
  
  return flat_coordinates


#########################################################################  
### METHODS FOR ALIGNING TWO SURFACES VIA THEIR CONFORMAL FLATTENINGS ###
######################################################################### 


def rotation(p, theta):
  #this is a helper method for the method getAlignedCoordinates below.  It rotates a given point in the plane about the origin by a given angle.
  return [cos(theta)*p[0]-sin(theta)*p[1],sin(theta)*p[0]+cos(theta)*p[1]]


def getAlignedCoordinates(animal_obj_0, animal_obj_1, theta):
  """ Calculates the vertex coordinates for the triangulation of Animal 1 aligned to the triangulation of Animal 0 by factoring
    through their respective conformal flattenings and applyling a rotation of angle theta.

    :Parameters:
      animal_obj_0/1 : animal objects, initialized with regular/flattened coordinates and triangulation set/updated
      theta : float with value between 0 and pi, an angle of rotation

    :Returns:
      list of triples of floats, specifying the x-, y-, and z-coordinates of the vertices of the triangulation of Animal 1 aligned to
      the triangulation of Animal 0
  """

  #store relevant parameters
  num_verts_0 = animal_obj_0.getNumVerts()
  regular_coordinates_0 = animal_obj_0.getRegularCoordinates()
  flat_coordinates_0 = animal_obj_0.getFlattenedCoordinates()
  flat_coordinates_0 = [f[:2] for f in flat_coordinates_0]
  triangles_0 = animal_obj_0.getTriangulation()
  num_verts_1 = animal_obj_1.getNumVerts()
  flat_coordinates_1 = animal_obj_1.getFlattenedCoordinates()
  flat_coordinates_1 = [f[:2] for f in flat_coordinates_1]

  #initialize return list
  aligned_coordinates_1 = []

  #iterate through the vertices of the triangulation of Animal 1
  for vertex in range(num_verts_1):

    #rotate the flattened coordinates of each such vertex by theta
    rotated_coordinate = rotation(flat_coordinates_1[vertex],theta)

    #initialize individual return values
    x, y, z = 0, 0, 0
    success = False

    #search through all the triangles in the triangulation of Animal 0 for one whose flattened coordinates contain
    #the rotated flattened coordinates of the current vertex in the triangulation of Animal 1
    for triangle in triangles_0:

      #extract flattened coordinates of the vertices of the given triangle
      x_0 = flat_coordinates_0[triangle[0]][0]
      x_1 = flat_coordinates_0[triangle[1]][0]
      x_2 = flat_coordinates_0[triangle[2]][0]
      y_0 = flat_coordinates_0[triangle[0]][1]
      y_1 = flat_coordinates_0[triangle[1]][1]
      y_2 = flat_coordinates_0[triangle[2]][1]

      #calculate barycentric coordinates for current vertex in current triangle
      lambda_0 = ((y_1-y_2)*(rotated_coordinate[0]-x_2)+(x_2-x_1)*(rotated_coordinate[1]-y_2)) / \
                ((y_1-y_2)*(x_0-x_2)+(x_2-x_1)*(y_0-y_2))
      lambda_1 = ((y_2-y_0)*(rotated_coordinate[0]-x_2)+(x_0-x_2)*(rotated_coordinate[1]-y_2)) / \
                ((y_1-y_2)*(x_0-x_2)+(x_2-x_1)*(y_0-y_2))
      lambda_2 = 1 - lambda_0 - lambda_1

      #if current triangle contains rotated flattened coordinates of current vertex, update return values using the barycentric
      #coordinates above and the regular coordinates of Animal 0
      if lambda_0 >= 0 and lambda_0 <= 1 and lambda_1 >=0 and lambda_1 <=1 and lambda_2 >= 0 and lambda_2 <= 1:
        location = triangle
        success = True
        x = lambda_0*regular_coordinates_0[location[0]][0] + \
            lambda_1*regular_coordinates_0[location[1]][0] + \
            lambda_2*regular_coordinates_0[location[2]][0]
        y = lambda_0*regular_coordinates_0[location[0]][1] + \
            lambda_1*regular_coordinates_0[location[1]][1] + \
            lambda_2*regular_coordinates_0[location[2]][1]
        z = lambda_0*regular_coordinates_0[location[0]][2] + \
            lambda_1*regular_coordinates_0[location[1]][2] + \
            lambda_2*regular_coordinates_0[location[2]][2]
        break

    #if no such triangle is found, update the return values with the coordinates of the closest vertex in Animal 0 to the current vertex
    if not success:
      closest_vertex = 0
      for candidate_vertex in range(num_verts_0):
        if linalg.norm(array(rotated_coordinate)-array(flat_coordinates_0[candidate_vertex])) < linalg.norm(array(rotated_coordinate)-array(flat_coordinates_0[closest_vertex])):
          closest_vertex = candidate_vertex
      x = regular_coordinates_0[closest_vertex][0]
      y = regular_coordinates_0[closest_vertex][1]
      z = regular_coordinates_0[closest_vertex][2]

    #append aligned coordinates to return list
    aligned_coordinates_1.append([x,y,z])

  return aligned_coordinates_1


def area(p, q, r):
  #this is a helper method for the distortionEnergy and computeOneCSD methods below. It calculates the
  #area of the triangle spanned by three points in R^2 or R^3.
  if len(p) == 2:
    p.append(0)
    q.append(0)
    r.append(0)
  x = []
  y = []
  for i in range(3):
    x.append(q[i]-p[i])
    y.append(r[i]-p[i])
  return 0.5*((x[1]*y[2]-x[2]*y[1])**2+(x[2]*y[0]-x[0]*y[2])**2+(x[0]*y[1]-x[1]*y[0])**2)**0.5


def distortionEnergy(animal_obj_0, animal_obj_1, rho):
  """ Calculates the elastic energy required to stretch the triangulation of Animal 0 onto the triangulation of Animal 1 
    via the conformal mapping obtained by factoring through their respective conformal flattenings and applyling a rotation 
    of angle rho.

    :Parameters:
      animal_obj_0/1 : animal objects, initialized with regular/flattened coordinates and triangulation set/updated
      rho : float with value between 0 and pi, an angle of rotation

    :Returns:
      float, specifying the elastic energy required to align the triangulation of Animal 1 that of Animal 0
  """

  #store relevant parameters
  num_verts = animal_obj_0.getNumVerts()
  regular_coordinates = animal_obj_0.getRegularCoordinates()
  aligned_coordinates = getAlignedCoordinates(animal_obj_1,animal_obj_0,rho)
  triangles = animal_obj_0.getTriangulation()

  #calculate four matrices whose entries correspond to pairs of vertices in the triangulation of Animal 0
  #with values given by (1) the number of triangles containing that pair of vertices, (2) the length of the
  #edge between them (if one exists) in the regular triangulation of Animal 0, (3) the length of the edge
  #between them (if one exists) in the triangulation of Animal 0 aligned to that of Animal 1 via the rotation
  #rho, and (4) the sum of the areas of the triangles in the regular triangulation of Animal 0 containing the
  #pair of vertices.
  incidence_matrix = [[[0 for k in range(4)] for j in range(num_verts)] for i in range(num_verts)]

  for triangle in triangles:
    sorted_triangle = sorted(triangle)
    u = sorted_triangle[0]
    v = sorted_triangle[1]
    w = sorted_triangle[2]

    incidence_matrix[v][u][0] += 1
    incidence_matrix[v][u][1] = linalg.norm(array(regular_coordinates[v])-array(regular_coordinates[u]))
    incidence_matrix[v][u][2] = linalg.norm(array(aligned_coordinates[v])-array(aligned_coordinates[u]))
    incidence_matrix[v][u][3] += area(regular_coordinates[u],regular_coordinates[v],regular_coordinates[w])

    incidence_matrix[w][u][0] += 1
    incidence_matrix[w][u][1] = linalg.norm(array(regular_coordinates[w])-array(regular_coordinates[u]))
    incidence_matrix[w][u][2] = linalg.norm(array(aligned_coordinates[w])-array(aligned_coordinates[u]))
    incidence_matrix[w][u][3] += area(regular_coordinates[u],regular_coordinates[v],regular_coordinates[w])

    incidence_matrix[w][v][0] += 1
    incidence_matrix[w][v][1] = linalg.norm(array(regular_coordinates[w])-array(regular_coordinates[v]))
    incidence_matrix[w][v][2] = linalg.norm(array(aligned_coordinates[w])-array(aligned_coordinates[v]))
    incidence_matrix[w][v][3] += area(regular_coordinates[u],regular_coordinates[v],regular_coordinates[w])

  #initialize the return value
  alignment_value = 0

  #sum the squares of the conformal stretching factors of the alignment over each edge in the triangulation
  for i in range(num_verts):
    for j in range(i):
      if incidence_matrix[i][j][0] == 2:
        alignment_value += (incidence_matrix[i][j][3]/3.0)*(incidence_matrix[i][j][2]/incidence_matrix[i][j][1]-1.0)**2

  return alignment_value**0.5

      
def symmetricDistortionEnergy(animal_obj_0, animal_obj_1, rho):
  """ Calculates the symmetric distortion energy required to stretch the triangulation of Animal 0 onto the 
    triangulation of Animal 1 and vice versa via the conformal mapping obtained by factoring through their 
    respective conformal flattenings and applyling a rotation of angle rho.

    :Parameters:
      animal_obj_0/1 : animal objects, initialized with regular/flattened coordinates and triangulation set/updated
      rho : float with value between 0 and pi, an angle of rotation

    :Returns:
      float, specifying the symmetric distortion energy required to align the triangulations of Animals 0 and 1
  """
  return distortionEnergy(animal_obj_0, animal_obj_1, rho) + distortionEnergy(animal_obj_1, animal_obj_0, -rho)


def optimalRotation(animal_obj_0,animal_obj_1):
  """ Calculates the optimal rotation of the unit disk that minimizes the symmetric distortion energy between 
    the triangulations of two animals

    :Parameters:
      animal_obj_0/1 : animal objects, initialized with regular/flattened coordinates and triangulation set/updated

    :Returns:
      float, specifying an angle between 0 and pi
  """

  #define a single variable function for a fixed pair of animals that takes an angle as input and outputs the
  #corresponding symmetric distortion energy
  def optimization_function(x):
    return symmetricDistortionEnergy(animal_obj_0,animal_obj_1,x)

  return minimize_scalar(optimization_function,bounds=(0,pi),method='Brent',tol=1.0).x


####################################################################################  
### METHODS FOR CALCULATING CONFORMAL SPATIOTEMPORAL DISTANCES BETWEEN HEAT MAPS ###
#################################################################################### 
  

def computeOneCSD(animal_obj_0, animal_obj_1, fullmode=False, outdir=None):
  """ Computes the Conformal Spatiotemporal Distance between the heatmaps of two animals

    :Parameters:
      animal_obj_0/1 : animal objects, initialized with regular/flattened coordinates and triangulation set/updated
      fullmode : Boolean, writes triangulations and their corresponding flattenings and alignments to .OFF files if True
      outdir : string, specifying directory to save .OFF files if fullmode is True

    :Returns:
      float, specifying the Conformal Spatiotemporal Distance between the heatmaps of two animals
  """

  #check that a directory is specified if fullmode is true
  if fullmode and outdir == None:
    throwError("Full mode requiers the path to output direcotry")

  #notify user of progress
  print("Measuring conformal spatiotemporal distance between heat maps of %s and %s..." % (animal_obj_0.getName(),animal_obj_1.getName()))

  #calculate the optimal rotation for aligning the triangulations of the two animals
  #theta = optimalRotation(animal_obj_0,animal_obj_1)
  theta = 0

  #store relevant parameters.  Note that we assume both animal observations have the same dimensions
  x_dim, y_dim = animal_obj_0.getDims()
  z_dim = getZDim(animal_obj_0)
  num_verts_0 = animal_obj_0.getNumVerts()
  regular_coordinates_0 = animal_obj_0.getRegularCoordinates()
  aligned_coordinates_0 = getAlignedCoordinates(animal_obj_1,animal_obj_0,theta)
  triangles_0 = animal_obj_0.getTriangulation()
  num_verts_1 = animal_obj_1.getNumVerts()
  regular_coordinates_1 = animal_obj_1.getRegularCoordinates()
  aligned_coordinates_1 = getAlignedCoordinates(animal_obj_0,animal_obj_1,-theta)
  triangles_1 = animal_obj_1.getTriangulation()

  #Save the triangulation data in .OFF files if fullmode is True
  if fullmode:
    write.writeOFF(animal_obj_0, regular_coordinates_0, outdir, "heatmap_%s_regular.off" % animal_obj_0.getName())
    write.writeOFF(animal_obj_1, regular_coordinates_1, outdir, "heatmap_%s_regular.off" % animal_obj_1.getName())
    write.writeOFF(animal_obj_0, animal_obj_0.getFlattenedCoordinates(), outdir, "heatmap_%s_flat.off" % animal_obj_0.getName())
    write.writeOFF(animal_obj_1, animal_obj_1.getFlattenedCoordinates(), outdir, "heatmap_%s_flat.off" % animal_obj_1.getName())
    write.writeOFF(animal_obj_0, aligned_coordinates_0, outdir, "heatmap_%s_aligned_to_%s.off" % (animal_obj_0.getName(),animal_obj_1.getName()))
    write.writeOFF(animal_obj_1, aligned_coordinates_1, outdir, "heatmap_%s_aligned_to_%s.off" % (animal_obj_1.getName(),animal_obj_0.getName()))

  #calculate the distance between the aligned surfaces 
  difference_val_0 = 0
  dA = 0
  for vertex in range(num_verts_1):
    for triangle in triangles_1:
      if vertex in triangle:
        dA += area(regular_coordinates_1[triangle[0]][0:2],regular_coordinates_1[triangle[1]][0:2],regular_coordinates_1[triangle[2]][0:2])/3.0
    difference_val_0 += dA*(aligned_coordinates_1[vertex][2]-regular_coordinates_1[vertex][2])**2

  difference_val_1 = 0
  dA = 0
  for vertex in range(num_verts_0):
    for triangle in triangles_0:
      if vertex in triangle:
        dA += area(regular_coordinates_0[triangle[0]][0:2],regular_coordinates_0[triangle[1]][0:2],regular_coordinates_0[triangle[2]][0:2])/3.0
    difference_val_1 += dA*(aligned_coordinates_0[vertex][2]-regular_coordinates_0[vertex][2])**2

  distance = (difference_val_0**0.5+difference_val_1**0.5)/(2*z_dim*x_dim*y_dim)

  #record distance in terminal
  print("LOG: distance  between aligned surfaces of %s and %s: %.3f" % (animal_obj_0.getName(), animal_obj_1.getName(), distance))

  return distance

  
def computeAllCSD(animal_list):
  """ Computes the Conformal Spatiotemporal Distances between the heatmaps of all pairs in list of animals

    :Parameters:
      animal_list : list of animal objects, initialized with regular/flattened coordinates and triangulation set/updated

    :Returns:
      2D array of floats, specifying the Conformal Spatiotemporal Distance between the heatmaps of each pair of animals in the input list
  """

  #initialize return array
  num_animals = len(animal_list)
  Dists = [['' for i in range(num_animals)] for j in range(num_animals)]

  #calculate the CSD between each pair of animals in the input list
  for i in range(num_animals):
    for j in range(i+1, num_animals):
      Dists[i][j] = computeOneCSD(animal_list[i],animal_list[j])
      
  return Dists
    
