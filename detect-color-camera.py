#
# * Copyright 2022 Vasoo Veerapen
# * vasoo (dot) veerapen (at) gmail (dot) com
# * https://www.linkedin.com/in/veerapen/
# *
# * This file is part of detect-color-camera.py
# * 
# * SHAREMIG is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import cv2
import numpy as np

#Find the average of a list of numbers
def average(lst): 
    return sum(lst) / len(lst) 

#Create an array with a number of zeros
def zero_maker(n):
    listofzeros = [0] * n
    return listofzeros

#Adjust the brightness to detect. Depends on your environment and camera
thresh_value = 100

#A list to store the number of points of interest detected over time
poi_queue = zero_maker(20)

#Capture from a webcam or RTSP stream
cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, source = cap.read()

    #Normalise the image
    norm_image = np.zeros(source.shape)
    norm_image = cv2.normalize(source, norm_image, 100, 255, cv2.NORM_MINMAX)

    #Remove noise from the source image
    median = cv2.medianBlur(norm_image, 15)

    # Convert BGR to HSV
    hsv_source = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)

    # Setup the "Lower red" mask
    lower_red = np.array([0,100,100])
    upper_red = np.array([5,255,255])
    red_mask_1 = cv2.inRange(hsv_source, lower_red, upper_red)

    # Setup the "Upper red" mask
    lower_red = np.array([160,100,100])
    upper_red = np.array([179,255,255])
    red_mask_2 = cv2.inRange(hsv_source, lower_red, upper_red)

    # Join both masks
    red_mask_final = red_mask_1 + red_mask_2

    # Bitwise-AND mask on the hsv_source image
    output_hsv = cv2.bitwise_and(hsv_source, hsv_source, mask = red_mask_final)

    # Blur the image to smoothen out bright areas
    blurred = cv2.GaussianBlur(output_hsv, (15, 15), 0)

    # Threshold the image to reveal bright areas
    thresh = cv2.threshold(blurred, thresh_value, 255, cv2.THRESH_BINARY)[1]

    # Split the image into HSV components
    h_chan, s_chan, v_chan = cv2.split(thresh)

    # Erode and dilate the gray scale component to remove small unwanted areas
    kernel = np.ones((3,3), np.uint8)
    v_chan = cv2.erode(v_chan, kernel, iterations=2)
    v_chan = cv2.dilate(v_chan, kernel, iterations=4)

    # Perform a connected component analysis
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(v_chan)

    # Create a mask to store the points of interest
    poi_mask = np.zeros(v_chan.shape, dtype="uint8")

    # Create a variable to store the number of points
    poi_count = 0

    # Loop over the labels
    for label in np.unique(labels):
        # Ignore the background label
        if label == 0:
            continue

        # Build the label mask and count the number of pixels in the area
        label_mask = np.zeros(v_chan.shape, dtype="uint8")
        label_mask[labels == label] = 255
        num_pixels = cv2.countNonZero(label_mask)

        # if the number of pixels in the component are sufficient in number then add the area to the point of interest mask
        if num_pixels > 50:
            poi_mask = cv2.add(poi_mask, label_mask)
            poi_count += 1


    #Convert the POI mask back to RGB before overlaying the source image
    poi_mask_back_to_rgb = cv2.cvtColor(poi_mask, cv2.COLOR_GRAY2RGB)

    # Find contours:
    contours, hierachy = cv2.findContours(poi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Mark contours
    for contour in contours:
        M = cv2.moments(contour)
        # Print center (debugging):
        #print("center X : '{}'".format(round(M['m10'] / M['m00'])))
        #print("center Y : '{}'".format(round(M['m01'] / M['m00'])))
        # Draw a circle based centered at centroid coordinates
        cv2.circle(source, (round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])), 20, (255, 255, 255), -1)

    #Remove the oldest value from the list and append the current value
    oldest_poi = poi_queue.pop(0)
    poi_queue.append(poi_count)
    #print ('POI queue:', poi_queue)

    #Calculate the average value and round up
    #This step helps to keep a constant POI value in event of warning LEDs turning on and off
    poi_average = average(poi_queue)
    poi_rounded = round(poi_average, 0)
    print ('Detected :', poi_rounded)

    cv2.imshow('Overlayed Source Image', source)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
