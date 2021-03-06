#!/usr/bin/python3

import glob
import os.path
from os import sendfile, popen
from PIL import Image
import sys
import math
import time
import threading
import subprocess as sp
from multiprocessing import Process, Manager, Value, Array
from pprint import pprint
from random import randint

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
YELLOW = (255, 255, 0)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKCYAN = '\033[96m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'


def colorDistance(c1, c2):
	return math.sqrt(sum([(c1[i] - c2[i])**2 for i in range(3)]))

def isEqualColor(c1, c2, epsilon = 30):
	return colorDistance(c1, c2) < epsilon


'''
# https://stackoverflow.com/questions/398299/looping-in-a-spiral
# thank u Tom J Nowell
def spural(size):
	width, height = size
	width -= 1
	height -= 1
	out = []
	
	x = y = 0
	dx = 0
	dy = -1

	for i in range(max(width, height)**2):
		if (-width/2 < x <= width/2) and (-height/2 < y <= height/2):
			out.append((x + width//2, y + height//2))
		if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
			dx, dy = -dy, dx
		x, y = x+dx, y+dy
	return out
'''
class Spural:
	def __init__(self, size = (0,0)):
		self.width, self.height = size
		self.width -= 1
		self.height -= 1
		self.x = self.y = 0
		self.dx = 0
		self.dy = -1
		self.i = 0
		self.i_max = max(self.width, self.height)**2;

	def __iter__(self):
		return self

	def __next__(self):
		output = None

		while output == None:
			if self.i >= self.i_max: raise StopIteration
			if (-self.width/2 < self.x <= self.width/2) and (-self.height/2 < self.y <= self.height/2):
				output = (self.x + self.width//2, self.y + self.height//2)
			if self.x == self.y or (self.x < 0 and self.x == -self.y) or (self.x > 0 and self.x == 1-self.y):
				self.dx, self.dy = -self.dy, self.dx
			self.x, self.y = self.x+self.dx, self.y+self.dy

			self.i += 1
		return output

class iSpural:
	def __init__(self, size = (0,0)):
		self.spural = Spural(size)
		
	def __iter__(self):
		return self

	def __next__(self):
		return (self.spural.i, self.spural.__next__())



def filterWhitePixels(epsilon = 30, image=None, bmp=None):
	whiteMask = [[False for j in range(image.size[1])] for i in range(image.size[0])]

	for col in range(image.size[0]):
		for row in range(image.size[1]):
			if (isEqualColor(bmp[col, row], WHITE, epsilon=epsilon)):
				bmp[col, row] = WHITE
				whiteMask[col][row] = True
			else:
				bmp[col, row] = BLACK
				whiteMask[col][row] = False

	return whiteMask, image


def findCircles(start=5, epsilon=20, whiteMask=None, thread_id=-1, thread_progress=None):
	circles = []
	unfiltered = []
	#spural_arr = spural((len(whiteMask),len(whiteMask[0]))) # spural
	spural_len = len(whiteMask) * len(whiteMask[0]);
	
	for i, coord in iSpural((len(whiteMask), len(whiteMask[0]))):
		(col, row) = coord

		# print percentage only once every N loops
		if (i % int(spural_len//100)) == 0:
			percentage = (100 * i) / spural_len
			thread_progress[thread_id] = (min(percentage * 10/7, 100), len(circles))
			#print("[" + imgPath + "] -> ", str((round( 100 *(100 * i) / len(spural_arr)) / 100)) + "%", "@", len(circles), "roundy bois")
			#print("=" * int(percentage) + ">" + "-" * int(100 - percentage))

			# stop checking the last 30% of the image, because it probably isn't there
			if (percentage > 70):
				#open(imgPath + "_coords.txt") as out:
				#	out.

				break

		if whiteMask[col][row]:
			radius_range = isCircle((col, row), start, epsilon=epsilon, whiteMask=whiteMask);
			if radius_range:
				circles.append([col, row, radius_range])
				#matchRing((col, row), radius_range[1], (0, 252, 0), epsilon=comparisonEpsilon)
				#matchRing((col, row), radius_range[0], (255, 0, 0), epsilon=comparisonEpsilon)
				#print("Found you a round boy:", [col, row, radius_range])

	return circles


# print("Segmentation fault.")
# exit(0) # easter egg


def isCircle(center, start = 5, epsilon = 10, limit = 100, whiteMask = None):
	start = start or 0
	epsilon = epsilon or 10

	radius_start = None
	radius_end = start

	WHITE_BOOL = True
	BLACK_BOOL = False

	if not matchRing(center, radius_end, WHITE_BOOL, whiteMask = whiteMask):
		return False

	# if we are in a white patch
	if matchRing(center, epsilon + start, WHITE_BOOL, whiteMask = whiteMask):
		return False

	while (radius_end <= limit):
		isBlack = matchRing(center, radius_end, BLACK_BOOL, whiteMask = whiteMask)
		isWhite = matchRing(center, radius_end, WHITE_BOOL, whiteMask = whiteMask)

		if (not (isWhite or isBlack) or isBlack ) and radius_start == None:
			radius_start = radius_end
			#radius_end = radius_start + epsilon
			#break

			if not matchRing(center, radius_start + epsilon, BLACK_BOOL, whiteMask = whiteMask):
				return False

		if isBlack:
			break

		radius_end += 1

	isOuterBlack = matchRing(center, radius_end, BLACK_BOOL, whiteMask = whiteMask)
	
	#if radius_end:
	#	print(radius_end - radius_start)

	if (isOuterBlack and radius_start and radius_start != radius_end and (radius_end - radius_start) <= epsilon):
		return (radius_start, radius_end)
	else:
		return False


def matchRing(center, r, color, epsilon=30, debug=False, bmp=None, whiteMask=None):
	if (r == 0):
		if (debug):
			bmp[center[0], center[1]] = color
			return
		else:
			return whiteMask[center[0]][center[1]] == color
	#increment = 0.001 # should be enough probably
	increment = 1 / r # should be exact  probably
	fi = 0
	while (fi < 2*math.pi):
		x = round(r * math.cos(fi) + center[0])
		y = round(r * math.sin(fi) + center[1])
		if (whiteMask and (x < 0 or y < 0 or x >= len(whiteMask) or y >= len(whiteMask[0]))):
			return False

		'''
x: (470, 505)
y: (970, 1007)

center: (487, 987)
r: ~35
		'''

		if (debug):
			bmp[x, y] = color
		else:
			#if (colorDistance(bmp[x, y], color) >= epsilon): # might be px[y,x] (doesn't matter though)
			# might be px[y,x] (doesn't matter though)
			if whiteMask[x][y] != color:
				return False
		fi += increment
	return True

def circleDistance(c1, c2):
	return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)


'''
# does nothing, used nowhere
# DO NOT DELETE

def findClusterCenters(circles, max_cluster_size=20):
	krogeci = []
	unvisited = [] # temporary array to avoid (problems with) mutating within the for-loop
	while len(circles) > 0:
		boi = circles[0] # reference point
		
		# find all points closer to boi than max_cluster_size
		# and calculate their avg coordinates
		avg = [0,0,0]
		i = 0
		for circle in circles:
			if (circleDistance(circle, boi) < max_cluster_size):
				avg[0] += circle[0]
				avg[1] += circle[1]
				avg[2] += (circle[2][0] + circle[2][1]) / 2
				i += 1
			else:
				unvisited.append(circle)
				
		avg[0] //= i
		avg[1] //= i
		avg[2] //= i

		print("average for boi "+str(boi)+": ", avg)
		krogeci.append(avg)

		circles = unvisited # repeat for unvisited circles
		unvisited = [] # zanimivo, dela
		
	print("Found krogeci:", krogeci, "lp")

	return krogeci
'''

def clusterize(circles, max_cluster_size=20):
	clusters = []
	# temporary array to avoid (problems with) mutating within the for-loop
	unvisited = []
	while len(circles) > 0:
		boi = circles[0]  # reference point

		# find all points closer to boi than max_cluster_size
		# and calculate their avg coordinates
		cluster = []
		i = 0
		for circle in circles:
			if (circleDistance(circle, boi) < max_cluster_size):
				cluster.append(circle)
				i += 1
			else:
				unvisited.append(circle)

		clusters.append(cluster)

		circles = unvisited  # repeat for unvisited circles
		unvisited = []  # zanimivo, dela
	return clusters

def findBestFromCluster(cluster):
	cluster.sort(key=lambda x: (x[2][1] - x[2][0])**2, reverse=False)
	return cluster[0]


def getRotation(upper, lower):
	# assumes upper is the center of rotation
	deltax = lower[0] - upper[0]
	deltay = lower[1] - upper[1]
	angle = math.atan2(deltax, deltay)
	return (2*math.pi - angle)*360/(2*math.pi)  # convert to degrees

def getScale(upper, lower, wantedDistance):
	dist = circleDistance(upper, lower)
	return wantedDistance / dist

def getTranslation(upper, wantedCoords):
	return (wantedCoords[0] - upper[0], wantedCoords[1] - upper[1])

#thread_output = []
#thread_progress = []
#thread_output = [(Value('i', 0), Value('i', 0), Value('i', 0)) for i in range(THREAD_COUNT)]
#thread_progress = [Value('i', 0) for i in range(THREAD_COUNT)]
def thread_function(file, thread_index, thread_output, thread_progress, debug=False):
	#global thread_output;

	#print("[" + str(thread_index) + "] opening youw fiwe uwu :3", file)
	img = Image.open(file).convert("RGB")
	pixels = img.load()
	#print("[" + str(thread_index) + "] DONE opening file (˘ε˘)")

	try:
		with open(file + "_bestBoiPosition.txt") as f:
			upper, lower = [pairApply(int, x.split(",")) for x in f.read().split("\n")]
			thread_output[thread_index] = (upper, lower)
			thread_progress[thread_index] = [100, "⌐■_■"]
			# reminder:
			#! Yᵒᵘ Oᶰˡʸ Lᶤᵛᵉ Oᶰᶜᵉ
		return;
	except Exception as e:
		pass #? ¯\_(ツ)_/¯

	#print("[" + str(thread_index) + "] genewating bwack awnd white mask >.<")
	whiteMask, img = filterWhitePixels(image=img, bmp=pixels)
	#print("[" + str(thread_index) + "] DONE genewating bwack awnd white mask (ᵕᴗ ᵕ⁎)")

	#//findPointOnImage((500,1000), bmp=pixels, color=GREEN, image=img, ringRadius=7)
	#//findPointOnImage((500,1000), bmp=pixels, color=RED, image=img, ringRadius=24)
	
	#//print("[" + str(thread_index) + "]")
	circles = findCircles(whiteMask=whiteMask, thread_id=thread_index, thread_progress=thread_progress)

	upper = None
	lower = None

	# findPointOnImage(circles[0], pixels, img, RED, 25) 
	
	img.save(file + "_filter.png")

	clusters = clusterize(circles)
	#//print(thread_index, len(clusters), [len(c) for c in clusters])
	prev = None

	#//print("clusters: ",[clusters[i][0] for i in range(len(clusters))])
	for cluster in clusters:
		bestBoi = findBestFromCluster(cluster)
		#//print(thread_index, bestBoi)
		if prev:
			if prev[1] > bestBoi[1]:
				lower = prev[:2] # ignore the linter error, lp
				upper = bestBoi[:2]
			else:
				lower = bestBoi[:2]
				upper = prev[:2]
		else:
			prev = bestBoi
		
		if debug:
			matchRing((bestBoi[0], bestBoi[1]), (bestBoi[2][0] + bestBoi[2][1])/2, RED, debug=True)
	if debug:
		img.save("out.png")

	if upper and lower:
		if not os.path.isfile(file + "_bestBoiPosition.txt"):
			with open(file + "_bestBoiPosition.txt", "w") as f:
				f.write(str(upper[0]) + "," + str(upper[1]) + "\n")
				f.write(str(upper[0]) + "," + str(upper[1]))

	#scaled = (upper / img.width, lower / img.height)
	#thread_output[thread_index] = scaled # returns the value
	thread_output[thread_index] = (upper, lower)  # returns the value


def move_cursor(y, x):
	print("\033[%d;%dH" % (y, x))

def print_padded(*str, width = 0):
	s = " ".join(str)
	padding_len = max(width - len(s), 0)
	padding = (' ' * padding_len) + '\n'
	print(s, end=padding)

def getTerminalDimensions():
	return pairApply(int, popen('stty size', 'r').read().split())

def thread_monitor(thread_progress, pool_count, pool_index):
	#global thread_progress
	#global pool_count
	#terminal_width = 85

	terminal_height, terminal_width = getTerminalDimensions()

	stars = ["✧", "*", "･", "ﾟ", "♡", ":", "☆", "ﾟ", ".", "･", "｡", "ﾟ"]
	sky = [stars[randint(0,len(stars)-1)] for i in range(terminal_width)]

	time.sleep(2) # wait for threads to initialise (just dont have a slow pc)
	_ = sp.call('clear', shell=True)

	while True:
		time.sleep(2)

		# update terminal sizerino
		terminal_height, terminal_width = getTerminalDimensions()

		move_cursor(0,0)

		print_padded("There is",str(pool_index.value + 1) + "/" + str(pool_count),"(-(-_-(-_(-_(-_-)_-)-_-)_-)_-) asian amogus", width=terminal_width)

		sum_done = 0
		round_boys = 0

		if len(thread_progress) == 0: continue
		
		
		for i, perc in enumerate(thread_progress):
			if not perc:
				print_padded("[%3d] sleeping... (◡ ω ◡)" % i, width=terminal_width)
				continue;

			sum_done += perc[0]
			try:
				round_boys += perc[1]
			except:
				pass #lp

			flooerd_perc = int(perc[0])
			print_padded("[%3d] found %5s roundy boiz %3d%% %s" % (
				i,
				str(perc[1]),
				flooerd_perc,
				("█" * round(flooerd_perc / 2.5)) + ("░" * round((100-flooerd_perc) / 2.5))
			), width=terminal_width)
			
		done_percent = int(sum_done // len(thread_progress))

		ber = " ʕっ•ᴥ•ʔっ"
		ber = bcolors.OKCYAN + ber + bcolors.ENDC
		ber_len = 10; # len(ber) = 17 due to unprintable characters
		UwU = "⊂(・▽・⊂)"
		###### 123456789
		UwU = bcolors.FAIL + UwU+ bcolors.ENDC
		UwU_len = 9 # len(UwU) = 14 due to unprintable characters

		print()
		# ᴇʙᴏʟᴀ
		#stars_progressed = ("".join([stars[x % len(stars)] for x in range(sum_done//2)]))
		max_space_count = max(terminal_width - ber_len - UwU_len, 0)
		space_count = int(max_space_count * ((100-done_percent)/100))
		stars_count = max_space_count - space_count
		animation = "".join(sky[:stars_count]) + ber + (" " * space_count) + UwU

		print_padded("[SUM] found %5d roundy boiz %3d%% %s" % (
			round_boys,
			done_percent,
			""#animation
		), width=terminal_width)
		print(animation)

		if done_percent >= 100:
			break

def avgOfPair(arrOfPairs):
	start = [0,0]
	for pair in arrOfPairs:
		if pair:
			start[0] += pair[0]
			start[1] += pair[1]
	start[1] /= len(arrOfPairs)
	start[0] /= len(arrOfPairs)
	return start

def findPointOnImage(pos, bmp=None, image=None, color = RED, ringRadius = 10):
	if not bmp:
		raise Exception("No bmp given òwó")
		return;

	if not image:
		raise Exception("No image given òwó")
		return;

	matchRing(pos, ringRadius, color, debug=True, bmp=bmp)

	# horizontal line
	for y in range(image.size[1]):
		bmp[pos[0], y] = color

	# vertical line
	for x in range(image.size[0]):
		bmp[x, pos[1]] = color;

	#print("I AM ERROR ÙwÚ, NOTICE ME SENPAII", pos, image.size)


def pairApply(f,p):
	return (f(p[0]), f(p[1]))


files = []
for f in glob.glob("*"):
	if f[-3:] == "jpg":
		files.append(f)

#files = files[:5]
#//files = ["2020-03-18.jpg"]
print(files)

# COMMENT

'''
for i in range(0, pool_count[1]):
	#print("DOING PAWTITION",i)
	pool_count[0] = i
	end = min((i+1)*THREAD_COUNT, len(files))
	partition = files[i*THREAD_COUNT : end]
	#print("GOING FWOM",i*THREAD_COUNT,"TO",end,"; PAWTITION:",partition)

	# spawn all threads
	threads = []
	for _,file in enumerate(partition):
		#thread_output.append(None); # fill array with None values to avoid IndexOutOfBoundsException
		#thread_progress.append((0,0))
		th = Process(target=thread_function, args=(file, thread_index, thread_output, thread_progress), daemon=True)
		th.start()
		threads.append(th)
		thread_index += 1;

	# await all threads
	for thread in threads:
		thread.join();
'''

THREAD_COUNT = 99
POOL_COUNT = math.ceil(len(files) / THREAD_COUNT)
results = []

with Manager() as manager:
	thread_index = 0
	thread_output = manager.list([None for i in range(len(files))]);
	thread_progress = manager.list([None for i in range(len(files))]);
	pool_index = manager.Value('i', 0)

	watchdog = Process(target=thread_monitor, args=(thread_progress, POOL_COUNT, pool_index), daemon=True) # daemons die when parent dies
	watchdog.start()
	
	for i in range(0, POOL_COUNT):
		#print("DOING PAWTITION",i)
		pool_index.value = i
		end = min((i+1)*THREAD_COUNT, len(files))
		partition = files[i*THREAD_COUNT : end]
		#print("GOING FWOM",i*THREAD_COUNT,"TO",end,"; PAWTITION:",partition)

		# spawn all threads
		threads = []
		for _,file in enumerate(partition):
			#thread_output.append(None); # fill array with None values to avoid IndexOutOfBoundsException
			#thread_progress.append((0,0))
			th = Process(target=thread_function, args=(file, thread_index, thread_output, thread_progress), daemon=True)
			th.start()
			threads.append(th)
			thread_index += 1;

		# await all threads
		for thread in threads:
			thread.join();

	watchdog.terminate() # watchdog must also terminate

	results = list(thread_output) # [x for x in thread_output]

# results now contains all the clusters
print("UwU aww my thweads awe done ଘ(੭ ˘ ᵕ˘)━☆ﾟ.*･｡ﾟᵕ꒳ᵕ~")
pprint(results)

uppers = [x[0] for x in results]
lowers = [x[1] for x in results]

average_upper = avgOfPair(uppers)
average_lower = avgOfPair(lowers)

distance = circleDistance(average_upper, average_lower)

images = []
sizes = []
for path in files:
	image = Image.open(path).convert("RGB")
	sizes.append(image.size)
	images.append(image)

avgSize = tuple(avgOfPair(sizes))

# rotate, scale and translate
for i,path in enumerate(files):
	upper = results[i][0]
	lower = results[i][1]
	image = images[i]

	scale = getScale(upper, lower, distance)
	new_size = (round(image.size[0] * scale), round(image.size[1] * scale))

	# adjust coords for new scale
	relative_upper_pos = (upper[0] / image.size[0], upper[1] / image.size[1])
	new_upper_pos = (relative_upper_pos[0] * new_size[0], relative_upper_pos[1] * new_size[1])
	relative_lower_pos = (lower[0] / image.size[0], lower[1] / image.size[1])
	new_lower_pos = (relative_lower_pos[0] * new_size[0], relative_lower_pos[1] * new_size[1])

	image = image.resize(new_size) # all problems begone (?)
	image = image.crop((0, 0) + avgSize) # tuple addition is concatenation
	image = image.rotate(getRotation(upper, lower),
	                    center=new_upper_pos, # per documentation, rotation is done first
	                    translate=getTranslation(new_upper_pos, average_upper))

	pixels = image.load()

	image.save(str(i) + "_adj_" + path.split(".")[0] + ".png")

print("all done (❀˘꒳˘)♡(˘꒳˘❀)")
