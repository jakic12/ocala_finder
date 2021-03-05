#!/usr/bin/python3

import glob
from PIL import Image
import sys
import math
import time
import threading
import subprocess as sp
from multiprocessing import Process

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
#	return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 + (c1[2] - c2[2])**2)

def isEqualColor(c1, c2, epsilon = 30):
	return colorDistance(c1, c2) < epsilon

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
	
def filterWhitePixels(epsilon = 20, image=None, bmp=None):
	# nobody ever reads this code
	for col in range(image.size[0]):
		for row in range(image.size[1]):
			if (isEqualColor(bmp[col, row], WHITE, epsilon=epsilon)):
				bmp[col, row] = WHITE
			else:
				# this comment will never be found
				bmp[col, row] = BLACK
	return image


def findCircles(start=5, epsilon=10, comparisonEpsilon=30, image=None, bmp=None, imgPath="nopath", thread_id=-1):
	circles = []
	unfiltered = []
	spural_arr = spural(image.size) # spural
	
	
	for i, coord in enumerate(spural_arr):
		(col, row) = coord

		# print percentage only once every N loops
		if (i % int(len(spural_arr)//100)) == 0:
			percentage = (100 * i) / len(spural_arr)
			thread_progress[thread_id] = (min(percentage * 10/7, 100), len(circles))
			#print("[" + imgPath + "] -> ", str((round( 100 *(100 * i) / len(spural_arr)) / 100)) + "%", "@", len(circles), "roundy bois")
			#print("=" * int(percentage) + ">" + "-" * int(100 - percentage))

			# stop checking the last 30% of the image, because it probably isn't there
			if (percentage > 70):
				#open(imgPath + "_coords.txt") as out:
				#	out.

				break

		if isEqualColor(bmp[col, row], WHITE):
			radius_range = isCircle((col, row), start, epsilon=epsilon, image=image, bmp=bmp)
			if radius_range:
				circles.append([col, row, radius_range])
				#matchRing((col, row), radius_range[1], (0, 252, 0), epsilon=comparisonEpsilon)
				#matchRing((col, row), radius_range[0], (255, 0, 0), epsilon=comparisonEpsilon)
				#print("Found you a round boy:", [col, row, radius_range])

	return circles


# print("Segmentation fault.")
# exit(0) # easter egg

def isCircle(center, start = 5, comparisonEpsilon = 30, epsilon = 10, limit = 100, image=None, bmp=None):
	start = start or 0
	epsilon = epsilon or 10

	radius_start = None
	radius_end = start

	if (not matchRing(center, radius_end, WHITE, epsilon=comparisonEpsilon, image=image, bmp=bmp)):
		return False

	while (radius_end <= limit):
		isBlack = matchRing(center, radius_end, BLACK, epsilon=comparisonEpsilon, image=image, bmp=bmp)
		isWhite = matchRing(center, radius_end, WHITE, epsilon=comparisonEpsilon, image=image, bmp=bmp)

		if (not (isWhite or isBlack) or isBlack ) and radius_start == None:
			radius_start = radius_end
			#radius_end = radius_start + epsilon
			#break

			if not matchRing(center, radius_start + epsilon, BLACK, epsilon=comparisonEpsilon, image=image, bmp=bmp):
				return False

		if isBlack:
			break

		radius_end += 1

	isOuterBlack = matchRing(center, radius_end, BLACK, epsilon=comparisonEpsilon, image=image, bmp=bmp)
	
	#if radius_end:
	#	print(radius_end - radius_start)

	if (isOuterBlack and radius_start and radius_start != radius_end and (radius_end - radius_start) <= epsilon):
		return (radius_start, radius_end)
	else:
		return False


def matchRing(center, r, color, epsilon=30, debug=False, image=None, bmp=None):
	if (r == 0):
		if (debug):
			bmp[center[0], center[1]] = color
			return
		else:
			return isEqualColor(bmp[center[0], center[1]], color, epsilon)
	#increment = 0.001 # should be enough probably
	increment = 1 / r # should be exact  probably
	fi = 0
	while (fi < 2*math.pi):
		x = round(r * math.cos(fi) + center[0])
		y = round(r * math.sin(fi) + center[1])
		if (x < 0 or y < 0 or x >= image.size[0] or y >= image.size[1]):
			return False

		if (debug):
			bmp[x, y] = color
		else:
			#if (colorDistance(bmp[x, y], color) >= epsilon): # might be px[y,x] (doesn't matter though)
			# might be px[y,x] (doesn't matter though)
			if not isEqualColor(bmp[x, y], color, epsilon):
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

thread_output = []
thread_progress = []
def thread_function(file, thread_index, debug=False):
	global thread_output;

	#print("[" + str(thread_index) + "] opening youw fiwe uwu :3", file)
	img = Image.open(file).convert("RGB")
	pixels = img.load()
	#print("[" + str(thread_index) + "] DONE opening file (˘ε˘)")
	
	#print("[" + str(thread_index) + "] genewating bwack awnd white mask >.<")
	img = filterWhitePixels(image=img, bmp=pixels)
	#print("[" + str(thread_index) + "] DONE genewating bwack awnd white mask (ᵕᴗ ᵕ⁎)")

	img.save(file + "_filter.png")
	
	#print("[" + str(thread_index) + "]")
	circles = findCircles(image=img, bmp=pixels, imgPath=file, thread_id=thread_index)

	upper = None
	lower = None

	clusters = clusterize(circles)
	prev = None
	for cluster in clusters:
		bestBoi = findBestFromCluster(cluster)
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

	#scaled = (upper / img.width, lower / img.height)
	#thread_output[thread_index] = scaled # returns the value
	thread_output[thread_index] = (upper, lower)  # returns the value


def move_cursor(y, x):
	print("\033[%d;%dH" % (y, x))

pool_count = []

def thread_monitor():
	global thread_progress

	time.sleep(3) # wait for threads to initialise (just dont have a slow pc)
	_ = sp.call('clear', shell=True)

	while True:
		#time.sleep(0.5)
		move_cursor(0,0)
		time.sleep(0.5)

		print("There is",str(pool_count[0]) + "/" + str(pool_count[1]),"(-(-_-(-_(-_(-_-)_-)-_-)_-)_-)-) asian amogus\n")

		sum_done = 0
		round_boys = 0

		if len(thread_progress) == 0: continue

		for i, perc in enumerate(thread_progress):
			if not perc:
				print("[%3d] sleeping... (◡ ω ◡)")
				continue;

			sum_done += perc[0]
			round_boys += perc[1]

			flooerd_perc = int(perc[0])
			print("[%3d] found %5d roundy boiz %3d%% %s" % (i, perc[1], flooerd_perc, ("█" * int(flooerd_perc // 2.5)) + ("░" * int((100 - flooerd_perc) // 2.5))))
			
		sum_done = int(sum_done // len(thread_progress))

		ber = " ʕっ•ᴥ•ʔっ"
		UwU = "(˘ε˘)"
		stars = "✧･ﾟ: *✧･ﾟ♡ *♡･ﾟ✧*: ･ﾟ✧☆ﾟ.*･｡ﾟ~"
		print()
		# ᴇʙᴏʟᴀ
		print("[SUM] found %5d roundy boiz %3d%% %s" % (round_boys, sum_done, ("".join([stars[x % len(stars)] for x in range(sum_done//2)])) + bcolors.OKCYAN + ber + bcolors.ENDC + (" " * (100//2 - (sum_done//2) - len(ber))) + bcolors.FAIL + UwU+ bcolors.ENDC))

		if sum_done >= 100:
			break

def avgOfPair(arrOfPairs):
	start = [0,0]
	for pair in arrOfPairs:
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

	matchRing(pos, ringRadius, color, debug=True, bmp=bmp, image=image)

	# horizontal line
	for y in range(image.size[1]):
		bmp[pos[0], y] = color

	# vertical line
	for x in range(image.size[0]):
		bmp[x, pos[1]] = color;

	print("I AM ERROR ÙwÚ, NOTICE ME SENPAII", pos, image.size)


def pairApply(f,p):
	return (f(p[0]), f(p[1]))


files = []
for f in glob.glob("*"):
	if f[-3:] == "jpg":
		files.append(f)

print(files)

# COMMENT

THREAD_COUNT = 4
pool_count = [0, math.ceil(len(files) / THREAD_COUNT)]

watchdog = threading.Thread(target=thread_monitor, args=(), daemon=True) # daemons die when parent dies
watchdog.start()
for i in range(0, pool_count[1]):
	#print("DOING PAWTITION",i)
	pool_count[0] = i
	end = min((i+1)*THREAD_COUNT, len(files))
	partition = files[i*THREAD_COUNT : end]
	#print("GOING FWOM",i*THREAD_COUNT,"TO",end,"; PAWTITION:",partition)

	# spawn all threads
	threads = []
	for thread_index,file in enumerate(partition):
		thread_output.append(None); # fill array with None values to avoid IndexOutOfBoundsException
		thread_progress.append((0,0))
		th = threading.Thread(target=thread_function, args=(file, thread_index), daemon=True)
		th.start()
		threads.append(th)

	# await all threads
	for thread in threads:
		thread.join();


# thread_output now contains all the clusters
print("UwU aww my thweads awe done ଘ(੭ ˘ ᵕ˘)━☆ﾟ.*･｡ﾟᵕ꒳ᵕ~")
pprint(thread_output)

uppers = [x[0] for x in thread_output]
lowers = [x[1] for x in thread_output]

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
	upper = thread_output[i][0]
	lower = thread_output[i][1]
	image = images[i]

	scale = getScale(upper, lower, distance)
	new_size = (round(image.size[0] * scale), round(image.size[1] * scale))

	# adjust coords for new scale
	relative_upper_pos = (upper[0] / image.size[0], upper[1] / image.size[1])
	new_upper_pos = (relative_upper_pos[0] * new_size[0], relative_upper_pos[1] * new_size[1])
	relative_lower_pos = (lower[0] / image.size[0], lower[1] / image.size[1])
	new_lower_pos = (relative_lower_pos[0] * new_size[0], relative_lower_pos[1] * new_size[1])

	#print("target upper:", average_upper, "target lower:", average_lower)
	#print("target distance:", distance)
	#print("scale is",scale)
	#print("new size is",new_size)

	image = image.resize(new_size) # all problems begone (?)
	image = image.crop((0, 0) + avgSize) # tuple addition is concatenation
	image = image.rotate(getRotation(upper, lower),
	                    center=new_upper_pos, # per documentation, rotation is done first
	                    translate=getTranslation(new_upper_pos, average_upper))

	pixels = image.load()
	
	#findPointOnImage(new_upper_pos, color=BLUE, bmp=pixels, image=image)
	#findPointOnImage((new_upper_pos[0], new_upper_pos[1] + distance), color=GREEN, bmp=pixels, image=image)
	#matchRing(new_upper_pos, distance, MAGENTA, image=image, bmp=pixels, debug=True)
	#findPointOnImage(average_upper, color=RED, bmp=pixels, image=image)
	#findPointOnImage(average_upper, color=CYAN, bmp=pixels, image=image)

	image.save(str(i) + "_adj_" + path.split(".")[0] + ".png")

print("all done (❀˘꒳˘)♡(˘꒳˘❀)")
