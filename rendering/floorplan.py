
from panda3d.egg import *
from panda3d.core import *
from rendering.obj2egg import ObjMaterial
from copy import deepcopy
import numpy as np
import cv2
import copy

from panda3d.core import loadPrcFileData

# Configure Panda3D to use an offscreen buffer
loadPrcFileData('', 'window-type offscreen')


def calcDistance(point_1, point_2):
    return pow(pow(point_1[0] - point_2[0], 2) + pow(point_1[1] - point_2[1], 2), 0.5)

def calcLineDim(line, lineWidth=-1):
    if abs(line[0][0] - line[1][0]) > abs(line[0][1] - line[1][1]):
        if lineWidth < 0 or abs(line[0][1] - line[1][1]) <= lineWidth:
            return 0
        pass
    elif abs(line[0][0] - line[1][0]) < abs(line[0][1] - line[1][1]):
        if lineWidth < 0 or abs(line[0][0] - line[1][0]) <= lineWidth:
            return 1
    else:
        return -1


def convert_egg_to_obj(egg_file, obj_file):
    import subprocess
    # Path to the Blender executable
    import platform
    if platform.system() == 'Windows':
        blender_path = r'C:/Program Files/Blender Foundation/Blender 4.4/blender.exe'
    else:
        blender_path = '/usr/local/bin/blender'

    # Blender script to import .egg and export .obj
    script = f"""
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.egg(filepath="{egg_file}")
bpy.ops.export_scene.obj(filepath="{obj_file}")
bpy.ops.wm.quit_blender()
"""
    script_file = 'convert_egg_to_obj.py'
    with open(script_file, 'w') as file:
        file.write(script)

    try:
        subprocess.run([blender_path, '--background', '--python', script_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting EGG to OBJ: {e}")


class FloorPlan():
  def __init__(self, filename):
      self.wallWidth = 0.005
      self.wallHeight = 0.3
      self.doorWidth = self.wallWidth
      self.doorHeight = self.wallHeight * 0.8
      self.windowWidth = self.wallWidth + 0.0001
      self.windowHeight = self.wallHeight * 0.5
      self.windowOffset = self.wallHeight * 0.4
      self.filename = filename

      self.floorMat = ObjMaterial()
      self.floorMat.name = 'floor'
      self.floorMat.put('map_Kd', self.filename + '.png')

      self.ceilingMat = ObjMaterial()
      self.ceilingMat.name = 'ceiling'
      self.ceilingMat.put('map_Kd', 'data/ceiling.jpg')

      self.wallMats = []
      wallMat_1 = ObjMaterial()
      wallMat_1.name = 'wall_1'
      wallMat_1.put('map_Kd', 'data/bedroom_wall.jpg')
      wallMat_2 = ObjMaterial()
      wallMat_2.name = 'wall_2'
      wallMat_2.put('map_Kd', 'data/kitchen_wall.jpg')
      wallMat_3 = ObjMaterial()
      wallMat_3.name = 'wall_3'
      wallMat_3.put('map_Kd', 'data/dining_wall.jpg')
      wallMat_4 = ObjMaterial()
      wallMat_4.name = 'wall_4'
      wallMat_4.put('map_Kd', 'data/bathroom_wall.jpg')
      wallMat_5 = ObjMaterial()
      wallMat_5.name = 'wall_5'
      wallMat_5.put('map_Kd', 'data/wall.jpg')

      self.wallMats.append(wallMat_3)
      self.wallMats.append(wallMat_2)
      self.wallMats.append(wallMat_1)
      self.wallMats.append(wallMat_4)
      self.wallMats.append(wallMat_4)
      self.wallMats.append(wallMat_1)
      self.wallMats.append(wallMat_5)
      self.wallMats.append(wallMat_2)
      self.wallMats.append(wallMat_4)
      self.wallMats.append(wallMat_5)
      self.wallMats.append(wallMat_5)

      self.doorMat = ObjMaterial()
      self.doorMat.name = 'door'
      self.doorMat.put('map_Kd', 'data/door.jpg')

      self.windowMat = ObjMaterial()
      self.windowMat.name = 'window'
      self.windowMat.put('map_Kd', 'data/window.jpg')

      self.iconNodes = {}
      self.iconNodes['cooking_counter'] = base.loader.loadModel('../rendering/data/cooking_counter.egg')
      self.iconNodes['bathtub'] = base.loader.loadModel('../rendering/data/bathtub.egg')
      self.iconNodes['toilet'] = base.loader.loadModel('../rendering/data/toilet.egg')
      self.iconNodes['washing_basin'] = base.loader.loadModel('../rendering/data/washing_basin.egg')
      
      self.read()  # Initialize walls, doors, icons, etc.

      return


    # return

  def read(self):
    if not self.filename.endswith('.txt'):
        floorplanFile = open(self.filename + '.txt', 'r')
    else:
        floorplanFile = open(self.filename, 'r')

    self.walls = []
    self.doors = []
    self.icons = []
    self.wallsInt = []
    for line in floorplanFile.readlines():
        line = line.strip()
        values = line.split('\t')
        if len(values) == 2:
            self.width = float(values[0])
            self.height = float(values[1])
            self.maxDim = max(self.width, self.height)
        elif len(values) == 6:
            wall = []
            for i in range(4):
                wall.append(float(values[i]))
            lineDim = calcLineDim(((wall[0], wall[1]), (wall[2], wall[3])))
            wall[lineDim], wall[2 + lineDim] = min(wall[lineDim], wall[2 + lineDim]), max(wall[lineDim], wall[2 + lineDim])
            wall[1 - lineDim] = wall[3 - lineDim] = (wall[1 - lineDim] + wall[3 - lineDim]) / 2
            wall.append(int(float(values[4])) - 1)
            wall.append(int(float(values[5])) - 1)
            for pointIndex in range(2):
                wall[pointIndex * 2 + 0] /= self.maxDim
                wall[pointIndex * 2 + 1] /= self.maxDim
            self.walls.append(wall)

            wallInt = []
            for i in range(4):
                wallInt.append(int(float(values[i])))
            wallInt[lineDim], wallInt[2 + lineDim] = min(wallInt[lineDim], wallInt[2 + lineDim]), max(wallInt[lineDim], wallInt[2 + lineDim])
            self.wallsInt.append(wallInt)
        elif len(values) == 7:
            item = []
            for i in range(4):
                item.append(float(values[i]))

            for pointIndex in range(2):
                item[pointIndex * 2 + 0] /= self.maxDim
                item[pointIndex * 2 + 1] /= self.maxDim

            if values[4] == 'door':
                self.doors.append(item)
            else:
                item.append(values[4])
                self.icons.append(item)
    return
 
  def generateFloor(self, data):
    floorGroup = EggGroup('floor')
    data.addChild(floorGroup)
    
    vp = EggVertexPool('floor_vertex')
    floorGroup.addChild(vp)


    exteriorWalls = []
    for wall in self.walls:
      if wall[4] == 10 or wall[5] == 10:
        exteriorWalls.append(copy.deepcopy(wall))


    exteriorOpenings = []
    for wall in exteriorWalls:
      lineDim = calcLineDim((wall[:2], wall[2:4]))
      for doorIndex, door in enumerate(self.doors):
        if calcLineDim((door[:2], door[2:4])) != lineDim:
          continue
        if door[lineDim] >= wall[lineDim] and door[2 + lineDim] <= wall[2 + lineDim] and abs(door[1 - lineDim] - wall[1 - lineDim]) <= self.wallWidth:
          exteriorOpenings.append(doorIndex)

    minDistance = 10000
    mainDoorIndex = -1
    for icon in self.icons:
      if icon[4] == 'entrance':
        for doorIndex in exteriorOpenings:
          door = self.doors[doorIndex]
          distance = pow(pow((door[0] + door[2]) / 2 - (icon[0] + icon[2]) / 2, 2) + pow((door[1] + door[3]) / 2 - (icon[1] + icon[3]) / 2, 2), 0.5)
          if distance < minDistance:
            minDistance = distance
            mainDoorIndex = doorIndex
        break

    self.startCameraPos = [0.5, -0.5, self.wallHeight * 0.5]
    self.startTarget = [0.5, 0.5, self.wallHeight * 0.5]
    if mainDoorIndex >= 0:
      mainDoor = self.doors[mainDoorIndex]
      lineDim = calcLineDim((mainDoor[:2], mainDoor[2:4]))
      fixedValue = (mainDoor[1 - lineDim] + mainDoor[3 - lineDim]) / 2
      imageSize = [self.width / self.maxDim, self.height / self.maxDim]
      side = int(fixedValue < imageSize[1 - lineDim] * 0.5) * 2 - 1
      self.startCameraPos[lineDim] = (mainDoor[lineDim] + mainDoor[2 + lineDim]) / 2
      self.startTarget[lineDim] = (mainDoor[lineDim] + mainDoor[2 + lineDim]) / 2
      self.startCameraPos[1 - lineDim] = fixedValue - 0.5 * side
      self.startTarget[1 - lineDim] = fixedValue + 0.5 * side
      
      self.startCameraPos[0] = 1 - self.startCameraPos[0]
      self.startTarget[0] = 1 - self.startTarget[0]
    
    newDoors = []
    self.windows = []
    for doorIndex, door in enumerate(self.doors):
      if doorIndex == mainDoorIndex or doorIndex not in exteriorOpenings:
        newDoors.append(door)
      else:
        self.windows.append(door)
    self.doors = newDoors


    exteriorWallLoops = []
    visitedMask = {}
    gap = 5.0 / self.maxDim
    for wallIndex, wall in enumerate(exteriorWalls):
      if wallIndex in visitedMask:
        continue
      visitedMask[wallIndex] = True
      exteriorWallLoop = []
      exteriorWallLoop.append(wall)
      for loopWall in exteriorWallLoop:
        for neighborWallIndex, neighborWall in enumerate(exteriorWalls):
          if neighborWallIndex in visitedMask:
            continue
          if calcDistance(neighborWall[:2], loopWall[2:4]) < gap:
            exteriorWallLoop.append(neighborWall)
            visitedMask[neighborWallIndex] = True
            break
          elif calcDistance(neighborWall[2:4], loopWall[2:4]) < gap:
            neighborWall[0], neighborWall[2] = neighborWall[2], neighborWall[0]
            neighborWall[1], neighborWall[3] = neighborWall[3], neighborWall[1]
            exteriorWallLoop.append(neighborWall)
            visitedMask[neighborWallIndex] = True
            break
      exteriorWallLoops.append(exteriorWallLoop)

    for exteriorWallLoop in exteriorWallLoops:
      poly = EggPolygon()
      floorGroup.addChild(poly)
      
      poly.setTexture(self.floorMat.getEggTexture())
      poly.setMaterial(self.floorMat.getEggMaterial())

      for wallIndex, wall in enumerate(exteriorWallLoop):
        if wallIndex == 0:
          v = EggVertex()
          v.setPos(Point3D(1 - wall[0], wall[1], 0))
          v.setUv(Point2D(wall[0] * self.maxDim / self.width, 1 - wall[1] * self.maxDim / self.height))
          poly.addVertex(vp.addVertex(v))
        else:
          v = EggVertex()
          v.setPos(Point3D(1 - (wall[0] + exteriorWallLoop[wallIndex - 1][2]) / 2, (wall[1] + exteriorWallLoop[wallIndex - 1][3]) / 2, 0))
          v.setUv(Point2D((wall[0] + exteriorWallLoop[wallIndex - 1][2]) / 2 * self.maxDim / self.width, 1 - (wall[1] + exteriorWallLoop[wallIndex - 1][3]) / 2 * self.maxDim / self.height))
          poly.addVertex(vp.addVertex(v))
        if wallIndex == len(exteriorWallLoop) - 1:
          v = EggVertex()
          v.setPos(Point3D(1 - wall[2], wall[3], 0))
          v.setUv(Point2D(wall[2] * self.maxDim / self.width, 1 - wall[3] * self.maxDim / self.height))
          poly.addVertex(vp.addVertex(v))

    ceilingGroup = EggGroup('ceiling')
    data.addChild(ceilingGroup)
    
    vp = EggVertexPool('ceiling_vertex')
    ceilingGroup.addChild(vp)

    for exteriorWallLoop in exteriorWallLoops:
      poly = EggPolygon()
      ceilingGroup.addChild(poly)
      
      poly.setTexture(self.ceilingMat.getEggTexture())
      poly.setMaterial(self.ceilingMat.getEggMaterial())

      for wallIndex, wall in enumerate(exteriorWallLoop):
        if wallIndex == 0:
          v = EggVertex()
          v.setPos(Point3D(1 - wall[0], wall[1], self.wallHeight))
          v.setUv(Point2D(wall[0], 1 - wall[1]))
          poly.addVertex(vp.addVertex(v))
        else:
          v = EggVertex()
          v.setPos(Point3D(1 - (wall[0] + exteriorWallLoop[wallIndex - 1][2]) / 2, (wall[1] + exteriorWallLoop[wallIndex - 1][3]) / 2, self.wallHeight))
          v.setUv(Point2D((wall[0] + exteriorWallLoop[wallIndex - 1][2]) / 2, 1 - (wall[1] + exteriorWallLoop[wallIndex - 1][3]) / 2))
          poly.addVertex(vp.addVertex(v))
        if wallIndex == len(exteriorWallLoop) - 1:
          v = EggVertex()
          v.setPos(Point3D(1 - wall[2], wall[3], self.wallHeight))
          v.setUv(Point2D(wall[2], 1 - wall[3]))
          poly.addVertex(vp.addVertex(v))
    return

  def generateWindows(self, data):
      windowsGroup = EggGroup('windows')
      data.addChild(windowsGroup)
      
      vp = EggVertexPool('window_vertex')
      windowsGroup.addChild(vp)

      for windowIndex, window in enumerate(self.windows):
          windowGroup = EggGroup('window_' + str(windowIndex))
          windowsGroup.addChild(windowGroup)
          
          lineDim = calcLineDim((window[:2], window[2:4]))
          
          if lineDim == 0:
              deltas = (0, self.windowWidth)
          else:
              deltas = (self.windowWidth, 0)

          poly = EggPolygon()
          windowGroup.addChild(poly)
          poly.setTexture(self.windowMat.getEggTexture())
          poly.setMaterial(self.windowMat.getEggMaterial())

          v = EggVertex()
          v.setPos(Point3D(1 - (window[0] - deltas[0]), window[1] - deltas[1], self.windowOffset))
          v.setUv(Point2D(0, 0))
          poly.addVertex(vp.addVertex(v))

          v = EggVertex()
          v.setPos(Point3D(1 - (window[2] - deltas[0]), window[3] - deltas[1], self.windowOffset))
          v.setUv(Point2D(1, 0))
          poly.addVertex(vp.addVertex(v))

          v = EggVertex()
          v.setPos(Point3D(1 - (window[2] - deltas[0]), window[3] - deltas[1], self.windowOffset + self.windowHeight))
          v.setUv(Point2D(1, 1))
          poly.addVertex(vp.addVertex(v))

          v = EggVertex()
          v.setPos(Point3D(1 - (window[0] - deltas[0]), window[1] - deltas[1], self.windowOffset + self.windowHeight))
          v.setUv(Point2D(0, 1))
          poly.addVertex(vp.addVertex(v))

          poly = EggPolygon()
          windowGroup.addChild(poly)
          poly.setTexture(self.windowMat.getEggTexture())
          poly.setMaterial(self.windowMat.getEggMaterial())

          v = EggVertex()
          v.setPos(Point3D(1 - (window[0] + deltas[0]), window[1] + deltas[1], self.windowOffset))
          v.setUv(Point2D(0, 0))
          poly.addVertex(vp.addVertex(v))

          v = EggVertex()
          v.setPos(Point3D(1 - (window[2] + deltas[0]), window[3] + deltas[1], self.windowOffset))
          v.setUv(Point2D(1, 0))
          poly.addVertex(vp.addVertex(v))

          v = EggVertex()
          v.setPos(Point3D(1 - (window[2] + deltas[0]), window[3] + deltas[1], self.windowOffset + self.windowHeight))
          v.setUv(Point2D(1, 1))
          poly.addVertex(vp.addVertex(v))

          v = EggVertex()
          v.setPos(Point3D(1 - (window[0] + deltas[0]), window[1] + deltas[1], self.windowOffset + self.windowHeight))
          v.setUv(Point2D(0, 1))
          poly.addVertex(vp.addVertex(v))
          
      return


  def generateIcons(self, scene):
      for icon in self.icons:
          if icon[4] not in self.iconNodes:
              continue
          node = deepcopy(self.iconNodes[icon[4]])
          node.setHpr(0, -90, 0)
          mins, maxs = node.getTightBounds()
          dimensions = Point3(maxs - mins)
          
          minDistances = [self.maxDim, self.maxDim, self.maxDim, self.maxDim]
          for wall in self.walls:
              lineDim = calcLineDim(((wall[0], wall[1]), (wall[2], wall[3])))
              if lineDim == -1:
                  continue
              if ((icon[lineDim] + icon[2 + lineDim]) / 2 - wall[lineDim]) * ((icon[lineDim] + icon[2 + lineDim]) / 2 - wall[2 + lineDim]) > 0:
                  continue
              side = int(wall[1 - lineDim] > (icon[1 - lineDim] + icon[3 - lineDim]) / 2)
              index = lineDim * 2 + side
              distance = abs(wall[1 - lineDim] - icon[1 - lineDim + side * 2])
              if distance < minDistances[index]:
                  minDistances[index] = distance

          orientation = 0
          if icon[4] in ['cooking_counter']:
              if icon[2] - icon[0] > icon[3] - icon[1]:
                  if minDistances[0] < minDistances[1]:
                      orientation = 0
                  else:
                      orientation = 1
              else:
                  if minDistances[2] < minDistances[3]:
                      orientation = 2
                  else:
                      orientation = 3
          elif icon[4] in ['toilet']:
              if icon[2] - icon[0] < icon[3] - icon[1]:
                  if minDistances[0] < minDistances[1]:
                      orientation = 0
                  else:
                      orientation = 1
              else:
                  if minDistances[2] < minDistances[3]:
                      orientation = 2
                  else:
                      orientation = 3
          elif icon[4] in ['washing_basin']:
              orientation = np.argmin(minDistances)

          if orientation == 1:
              node.setH(180)
          elif orientation == 2:
              node.setH(90)
          elif orientation == 3:
              node.setH(270)
          if icon[4] == 'washing_basin':
              node.setH(90 + node.getH())
          mins, maxs = node.getTightBounds()
          dimensions = Point3(maxs - mins)

          scaleX = (icon[2] - icon[0]) / dimensions.getX()
          scaleY = (icon[3] - icon[1]) / dimensions.getY()
          scaleZ = max(scaleX, scaleY)
          node.setScale(scaleX, scaleY, scaleZ)
          node.setPos(1 - icon[0] - maxs.getX() * scaleX, icon[1] - mins.getY() * scaleY, -mins.getZ() * scaleZ)
          
          node.setTwoSided(True)
          node.reparentTo(scene)
      return


  def generateWalls(self, data):
      wallsGroup = EggGroup('walls')
      data.addChild(wallsGroup)

      vp = EggVertexPool('wall_vertex')
      data.addChild(vp)

      for wallIndex, wall in enumerate(self.walls):
          wallGroup = EggGroup('wall')
          wallsGroup.addChild(wallGroup)
          lineDim = calcLineDim((wall[:2], wall[2:4]))

          if lineDim == 0:
              deltas = (0, self.wallWidth)
          else:
              deltas = (self.wallWidth, 0)

          poly = EggPolygon()
          wallGroup.addChild(poly)

          if lineDim == 0:
              poly.setTexture(self.wallMats[wall[4]].getEggTexture())
              poly.setMaterial(self.wallMats[wall[4]].getEggMaterial())
          else:
              poly.setTexture(self.wallMats[wall[5]].getEggTexture())
              poly.setMaterial(self.wallMats[wall[5]].getEggMaterial())

          values = [wall[lineDim] - self.wallWidth + 0.0001, wall[2 + lineDim] + self.wallWidth - 0.0001]
          for door in self.doors:
              if calcLineDim((door[:2], door[2:4])) != lineDim:
                  continue
              if door[lineDim] >= wall[lineDim] and door[2 + lineDim] <= wall[2 + lineDim] and abs(door[1 - lineDim] - wall[1 - lineDim]) <= self.wallWidth:
                  values.append(door[lineDim])
                  values.append(door[2 + lineDim])
          values.sort()

          fixedValue = (wall[1 - lineDim] + wall[3 - lineDim]) / 2
          for valueIndex, value in enumerate(values):
              if valueIndex % 2 == 0 and valueIndex > 0:
                  v = EggVertex()
                  if lineDim == 0:
                      v.setPos(Point3D(1 - (value - deltas[0]), fixedValue - deltas[1], self.doorHeight))
                  else:
                      v.setPos(Point3D(1 - (fixedValue - deltas[0]), value - deltas[1], self.doorHeight))
                  v.setUv(Point2D(self.doorHeight / self.wallHeight, (value - wall[lineDim]) / (wall[2 + lineDim] - wall[lineDim])))
                  poly.addVertex(vp.addVertex(v))

              v = EggVertex()
              if lineDim == 0:
                  v.setPos(Point3D(1 - (value - deltas[0]), fixedValue - deltas[1], 0))
              else:
                  v.setPos(Point3D(1 - (fixedValue - deltas[0]), value - deltas[1], 0))
              v.setUv(Point2D(0, (value - wall[lineDim]) / (wall[2 + lineDim] - wall[lineDim])))
              poly.addVertex(vp.addVertex(v))

              if valueIndex % 2 == 1 and valueIndex + 1 < len(values):
                  v = EggVertex()
                  if lineDim == 0:
                      v.setPos(Point3D(1 - (value - deltas[0]), fixedValue - deltas[1], self.doorHeight))
                  else:
                      v.setPos(Point3D(1 - (fixedValue - deltas[0]), value - deltas[1], self.doorHeight))
                  v.setUv(Point2D(self.doorHeight / self.wallHeight, (value - wall[lineDim]) / (wall[2 + lineDim] - wall[lineDim])))
                  poly.addVertex(vp.addVertex(v))

          v = EggVertex()
          if lineDim == 0:
              v.setPos(Point3D(1 - (values[len(values) - 1] - deltas[0]), fixedValue - deltas[1], self.wallHeight))
          else:
              v.setPos(Point3D(1 - (fixedValue - deltas[0]), values[len(values) - 1] - deltas[1], self.wallHeight))
          v.setUv(Point2D(1, 1))
          poly.addVertex(vp.addVertex(v))

          v = EggVertex()
          if lineDim == 0:
              v.setPos(Point3D(1 - (values[0] - deltas[0]), fixedValue - deltas[1], self.wallHeight))
          else:
              v.setPos(Point3D(1 - (fixedValue - deltas[0]), values[0] - deltas[1], self.wallHeight))
          v.setUv(Point2D(1, 0))
          poly.addVertex(vp.addVertex(v))

          poly = EggPolygon()
          wallGroup.addChild(poly)
          if lineDim == 0:
              poly.setTexture(self.wallMats[wall[5]].getEggTexture())
              poly.setMaterial(self.wallMats[wall[5]].getEggMaterial())
          else:
              poly.setTexture(self.wallMats[wall[4]].getEggTexture())
              poly.setMaterial(self.wallMats[wall[4]].getEggMaterial())

          for valueIndex, value in enumerate(values):
              if valueIndex % 2 == 0 and valueIndex > 0:
                  v = EggVertex()
                  if lineDim == 0:
                      v.setPos(Point3D(1 - (value + deltas[0]), fixedValue + deltas[1], self.doorHeight))
                  else:
                      v.setPos(Point3D(1 - (fixedValue + deltas[0]), value + deltas[1], self.doorHeight))
                  v.setUv(Point2D(self.doorHeight / self.wallHeight, (value - wall[lineDim]) / (wall[2 + lineDim] - wall[lineDim])))
                  poly.addVertex(vp.addVertex(v))

              v = EggVertex()
              if lineDim == 0:
                  v.setPos(Point3D(1 - (value + deltas[0]), fixedValue + deltas[1], 0))
              else:
                  v.setPos(Point3D(1 - (fixedValue + deltas[0]), value + deltas[1], 0))
              v.setUv(Point2D(0, (value - wall[lineDim]) / (wall[2 + lineDim] - wall[lineDim])))
              poly.addVertex(vp.addVertex(v))

              if valueIndex % 2 == 1 and valueIndex + 1 < len(values):
                  v = EggVertex()
                  if lineDim == 0:
                      v.setPos(Point3D(1 - (value + deltas[0]), fixedValue + deltas[1], self.doorHeight))
                  else:
                      v.setPos(Point3D(1 - (fixedValue + deltas[0]), value + deltas[1], self.doorHeight))
                  v.setUv(Point2D(self.doorHeight / self.wallHeight, (value - wall[lineDim]) / (wall[2 + lineDim] - wall[lineDim])))
                  poly.addVertex(vp.addVertex(v))

          v = EggVertex()
          if lineDim == 0:
              v.setPos(Point3D(1 - (values[len(values) - 1] + deltas[0]), fixedValue + deltas[1], self.wallHeight))
          else:
              v.setPos(Point3D(1 - (fixedValue + deltas[0]), values[len(values) - 1] + deltas[1], self.wallHeight))
          v.setUv(Point2D(1, 1))
          poly.addVertex(vp.addVertex(v))

          v = EggVertex()
          if lineDim == 0:
              v.setPos(Point3D(1 - (values[0] + deltas[0]), fixedValue + deltas[1], self.wallHeight))
          else:
              v.setPos(Point3D(1 - (fixedValue + deltas[0]), values[0] + deltas[1], self.wallHeight))
          v.setUv(Point2D(1, 0))
          poly.addVertex(vp.addVertex(v))

          if lineDim == 0:
              poly = EggPolygon()
              wallGroup.addChild(poly)
              poly.setTexture(self.wallMats[10].getEggTexture())
              poly.setMaterial(self.wallMats[10].getEggMaterial())

              v = EggVertex()
              v.setPos(Point3D(1 - values[0], fixedValue - deltas[1], 0))
              v.setUv(Point2D(0, 0))
              poly.addVertex(vp.addVertex(v))

              v = EggVertex()
              v.setPos(Point3D(1 - values[0], fixedValue - deltas[1], self.wallHeight))
              v.setUv(Point2D(0, 1))
              poly.addVertex(vp.addVertex(v))

              v = EggVertex()
              v.setPos(Point3D(1 - values[0], fixedValue + deltas[1], self.wallHeight))
              v.setUv(Point2D(1, 1))
              poly.addVertex(vp.addVertex(v))

              v = EggVertex()
              v.setPos(Point3D(1 - values[0], fixedValue + deltas[1], 0))
              v.setUv(Point2D(1, 0))
              poly.addVertex(vp.addVertex(v))

              poly = EggPolygon()
              wallGroup.addChild(poly)
              poly.setTexture(self.wallMats[10].getEggTexture())
              poly.setMaterial(self.wallMats[10].getEggMaterial())

              v = EggVertex()
              v.setPos(Point3D(1 - values[0], fixedValue - deltas[1], self.wallHeight))
              v.setUv(Point2D(0, 0))
              poly.addVertex(vp.addVertex(v))

              v = EggVertex()
              v.setPos(Point3D(1 - values[len(values) - 1], fixedValue - deltas[1], self.wallHeight))
              v.setUv(Point2D(0, 1))
              poly.addVertex(vp.addVertex(v))

              v = EggVertex()
              v.setPos(Point3D(1 - values[len(values) - 1], fixedValue + deltas[1], self.wallHeight))
              v.setUv(Point2D(1, 1))
              poly.addVertex(vp.addVertex(v))

              v = EggVertex()
              v.setPos(Point3D(1 - values[0], fixedValue + deltas[1], self.wallHeight))
              v.setUv(Point2D(1, 0))
              poly.addVertex(vp.addVertex(v))

              poly = EggPolygon()
              wallGroup.addChild(poly)
              poly.setTexture(self.wallMats[10].getEggTexture())
              poly.setMaterial(self.wallMats[10].getEggMaterial())

              v = EggVertex()
              v.setPos(Point3D(1 - values[len(values) - 1], fixedValue - deltas[1], self.wallHeight))
              v.setUv(Point2D(0, 0))
              poly.addVertex(vp.addVertex(v))

              v = EggVertex()
              v.setPos(Point3D(1 - values[len(values) - 1], fixedValue - deltas[1], 0))
              v.setUv(Point2D(0, 1))
              poly.addVertex(vp.addVertex(v))

              v = EggVertex()
              v.setPos(Point3D(1 - values[len(values) - 1], fixedValue + deltas[1], 0))
              v.setUv(Point2D(1, 1))
              poly.addVertex(vp.addVertex(v))

              v = EggVertex()
              v.setPos(Point3D(1 - values[len(values) - 1], fixedValue + deltas[1], self.wallHeight))
              v.setUv(Point2D(1, 0))
              poly.addVertex(vp.addVertex(v))
          else:
              poly = EggPolygon()
              wallGroup.addChild(poly)
              poly.setTexture(self.wallMats[10].getEggTexture())
              poly.setMaterial(self.wallMats[10].getEggMaterial())

              v = EggVertex()
              v.setPos(Point3D(1 - (fixedValue - deltas[0]), values[0], 0))
              v.setUv(Point2D(0, 0))
              poly.addVertex(vp.addVertex(v))

              v = EggVertex()
              v.setPos(Point3D(1 - (fixedValue - deltas[0]), values[0], self.wallHeight))
              v.setUv(Point2D(0, 1))
              poly.addVertex(vp.addVertex(v))

              v = EggVertex()
              v.setPos(Point3D(1 - (fixedValue + deltas[0]), values[0], self.wallHeight))
              v.setUv(Point2D(1, 1))
              poly.addVertex(vp.addVertex(v))

              v = EggVertex()
              v.setPos(Point3D(1 - (fixedValue + deltas[0]), values[0], 0))
              v.setUv(Point2D(1, 0))
              poly.addVertex(vp.addVertex(v))

              poly = EggPolygon()
              wallGroup.addChild(poly)
              poly.setTexture(self.wallMats[10].getEggTexture())
              poly.setMaterial(self.wallMats[10].getEggMaterial())

              v = EggVertex()
              v.setPos(Point3D(1 - (fixedValue - deltas[0]), values[0], self.wallHeight))
              v.setUv(Point2D(0, 0))
              poly.addVertex(vp.addVertex(v))

              v = EggVertex()
              v.setPos(Point3D(1 - (fixedValue - deltas[0]), values[len(values) - 1], self.wallHeight))
              v.setUv(Point2D(0, 1))
              poly.addVertex(vp.addVertex(v))

              v = EggVertex()
              v.setPos(Point3D(1 - (fixedValue + deltas[0]), values[len(values) - 1], self.wallHeight))
              v.setUv(Point2D(1, 1))
              poly.addVertex(vp.addVertex(v))

              v = EggVertex()
              v.setPos(Point3D(1 - (fixedValue + deltas[0]), values[0], self.wallHeight))
              v.setUv(Point2D(1, 0))
              poly.addVertex(vp.addVertex(v))

              poly = EggPolygon()
              wallGroup.addChild(poly)
              poly.setTexture(self.wallMats[10].getEggTexture())
              poly.setMaterial(self.wallMats[10].getEggMaterial())

              v = EggVertex()
              v.setPos(Point3D(1 - (fixedValue - deltas[0]), values[len(values) - 1], self.wallHeight))
              v.setUv(Point2D(0, 0))
              poly.addVertex(vp.addVertex(v))

              v = EggVertex()
              v.setPos(Point3D(1 - (fixedValue - deltas[0]), values[len(values) - 1], 0))
              v.setUv(Point2D(0, 1))
              poly.addVertex(vp.addVertex(v))

              v = EggVertex()
              v.setPos(Point3D(1 - (fixedValue + deltas[0]), values[len(values) - 1], 0))
              v.setUv(Point2D(1, 1))
              poly.addVertex(vp.addVertex(v))

              v = EggVertex()
              v.setPos(Point3D(1 - (fixedValue + deltas[0]), values[len(values) - 1], self.wallHeight))
              v.setUv(Point2D(1, 0))
              poly.addVertex(vp.addVertex(v))
      return

  def generateDoors(self, data):
    doorsGroup = EggGroup('doors')
    data.addChild(doorsGroup)
    
    vp = EggVertexPool('door_vertex')
    doorsGroup.addChild(vp)

    for doorIndex, door in enumerate(self.doors):
        doorGroup = EggGroup('door_' + str(doorIndex))
        doorsGroup.addChild(doorGroup)
        
        lineDim = calcLineDim((door[:2], door[2:4]))
        
        if lineDim == 0:
            deltas = (0, self.doorWidth)
        else:
            deltas = (self.doorWidth, 0)

        poly = EggPolygon()
        doorGroup.addChild(poly)
        poly.setTexture(self.doorMat.getEggTexture())
        poly.setMaterial(self.doorMat.getEggMaterial())

        v = EggVertex()
        v.setPos(Point3D(1 - (door[0] - deltas[0]), door[1] - deltas[1], 0))
        v.setUv(Point2D(0, 0))
        poly.addVertex(vp.addVertex(v))

        v = EggVertex()
        v.setPos(Point3D(1 - (door[2] - deltas[0]), door[3] - deltas[1], 0))
        v.setUv(Point2D(1, 0))
        poly.addVertex(vp.addVertex(v))

        v = EggVertex()
        v.setPos(Point3D(1 - (door[2] - deltas[0]), door[3] - deltas[1], self.doorHeight))
        v.setUv(Point2D(1, 1))
        poly.addVertex(vp.addVertex(v))

        v = EggVertex()
        v.setPos(Point3D(1 - (door[0] - deltas[0]), door[1] - deltas[1], self.doorHeight))
        v.setUv(Point2D(0, 1))
        poly.addVertex(vp.addVertex(v))


        poly = EggPolygon()
        doorGroup.addChild(poly)
        poly.setTexture(self.doorMat.getEggTexture())
        poly.setMaterial(self.doorMat.getEggMaterial())

        v = EggVertex()
        v.setPos(Point3D(1 - (door[0] + deltas[0]), door[1] + deltas[1], 0))
        v.setUv(Point2D(0, 0))
        poly.addVertex(vp.addVertex(v))

        v = EggVertex()
        v.setPos(Point3D(1 - (door[2] + deltas[0]), door[3] + deltas[1], 0))
        v.setUv(Point2D(1, 0))
        poly.addVertex(vp.addVertex(v))

        v = EggVertex()
        v.setPos(Point3D(1 - (door[2] + deltas[0]), door[3] + deltas[1], self.doorHeight))
        v.setUv(Point2D(1, 1))
        poly.addVertex(vp.addVertex(v))

        v = EggVertex()
        v.setPos(Point3D(1 - (door[0] + deltas[0]), door[1] + deltas[1], self.doorHeight))
        v.setUv(Point2D(0, 1))
        poly.addVertex(vp.addVertex(v))
        
    return


  def generateEggModel(self, output_prefix=None):
      data = EggData()
      model = EggGroup('model')
      data.addChild(model)
      self.generateFloor(model)
      self.generateWalls(model)
      self.generateDoors(model)
      self.generateWindows(model)
      
      egg_output_path = output_prefix + "floorplan.egg"
      data.writeEgg(Filename(egg_output_path))
      
      # Convert .egg to .obj
      convert_egg_to_obj(egg_output_path, output_prefix + "floorplan.obj")
      
      scene = NodePath(loadEggData(data))
      self.generateIcons(scene)
      return scene
  
  def segmentRooms(self):
      wallMask = np.ones((self.height, self.width), np.uint8) * 255
      for wall in self.wallsInt:
          lineDim = calcLineDim(((wall[0], wall[1]), (wall[2], wall[3])))
          if lineDim == 0:
              wallMask[wall[1], wall[0]:wall[2] + 1] = 0
          else:
              wallMask[wall[1]:wall[3] + 1, wall[0]] = 0
      cv2.imwrite('test/walls.png', wallMask)
      
      numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(wallMask, 4)
      print(numLabels)
      print(labels.shape)
      print(stats.shape)
      print(centroids.shape)
      cv2.imwrite('test/rooms.png', labels)
