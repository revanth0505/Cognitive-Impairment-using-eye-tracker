import streamlit as st
import pandas as pd
import random
import os
import numpy as np
import numpy
from statistics import mean
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
def remove_missing(x, y, time, missing):
    mx = np.array(x==missing, dtype=int)
    my = np.array(y==missing, dtype=int)
    x = x[(mx+my) != 2]
    y = y[(mx+my) != 2]
    time = time[(mx+my) != 2]
    return x, y, time
def nanmean(values):
    non_nan_values = [x for x in values if not np.isnan(x)]
    return np.mean(non_nan_values)
def blink_detection(x, y, time, missing=0.0, minlen=10):
	
	"""Detects blinks, defined as a period of missing data that lasts for at
	least a minimal amount of samples
	
	arguments
	x		-	numpy array of x positions
	y		-	numpy array of y positions
	time		-	numpy array of EyeTribe timestamps
	keyword arguments
	missing	-	value to be used for missing data (default = 0.0)
	minlen	-	integer indicating the minimal amount of consecutive
				missing samples
	
	returns
	Sblk, Eblk
				Sblk	-	list of lists, each containing [starttime]
				Eblk	-	list of lists, each containing [starttime, endtime, duration]
	"""
	
	# empty list to contain data
	Sblk = []
	Eblk = []
	
	# check where the missing samples are
	mx = numpy.array(x==missing, dtype=int)
	my = numpy.array(y==missing, dtype=int)
	miss = numpy.array((mx+my) == 2, dtype=int)
	
	# check where the starts and ends are (+1 to counteract shift to left)
	diff = numpy.diff(miss)
	starts = numpy.where(diff==1)[0] + 1
	ends = numpy.where(diff==-1)[0] + 1
	
	# compile blink starts and ends
	for i in range(len(starts)):
		# get starting index
		s = starts[i]
		# get ending index
		if i < len(ends):
			e = ends[i]
		elif len(ends) > 0:
			e = ends[-1]
		else:
			e = -1
		# append only if the duration in samples is equal to or greater than
		# the minimal duration
		if e-s >= minlen:
			# add starting time
			Sblk.append([time[s]])
			# add ending time
			Eblk.append([time[s],time[e],time[e]-time[s]])
	
	return Sblk, Eblk
def fixation_detection(x, y, time, missing=0.0, maxdist=25, mindur=50):
	
	"""Detects fixations, defined as consecutive samples with an inter-sample
	distance of less than a set amount of pixels (disregarding missing data)
	
	arguments
	x		-	numpy array of x positions
	y		-	numpy array of y positions
	time		-	numpy array of EyeTribe timestamps
	keyword arguments
	missing	-	value to be used for missing data (default = 0.0)
	maxdist	-	maximal inter sample distance in pixels (default = 25)
	mindur	-	minimal duration of a fixation in milliseconds; detected
				fixation cadidates will be disregarded if they are below
				this duration (default = 100)
	
	returns
	Sfix, Efix
				Sfix	-	list of lists, each containing [starttime]
				Efix	-	list of lists, each containing [starttime, endtime, duration, endx, endy]
	"""

	x, y, time = remove_missing(x, y, time, missing)

	# empty list to contain data
	Sfix = []
	Efix = []
	
	# loop through all coordinates
	si = 0
	fixstart = False
	for i in range(1,len(x)):
		# calculate Euclidean distance from the current fixation coordinate
		# to the next coordinate
		squared_distance = ((x[si]-x[i])**2 + (y[si]-y[i])**2)
		dist = 0.0
		if squared_distance > 0:
			dist = squared_distance**0.5
		# check if the next coordinate is below maximal distance
		if dist <= maxdist and not fixstart:
			# start a new fixation
			si = 0 + i
			fixstart = True
			Sfix.append([time[i]])
		elif dist > maxdist and fixstart:
			# end the current fixation
			fixstart = False
			# only store the fixation if the duration is ok
			if time[i-1]-Sfix[-1][0] >= mindur:
				Efix.append([Sfix[-1][0], time[i-1], time[i-1]-Sfix[-1][0], x[si], y[si]])
			# delete the last fixation start if it was too short
			else:
				Sfix.pop(-1)
			si = 0 + i
		elif not fixstart:
			si += 1
	#add last fixation end (we can lose it if dist > maxdist is false for the last point)
	if len(Sfix) > len(Efix):
		Efix.append([Sfix[-1][0], time[len(x)-1], time[len(x)-1]-Sfix[-1][0], x[si], y[si]])
	return Sfix, Efix
def saccade_detection(x, y, time, missing=0.0, minlen=5, maxvel=40, maxacc=340):
	
	"""Detects saccades, defined as consecutive samples with an inter-sample
	velocity of over a velocity threshold or an acceleration threshold
	
	arguments
	x		-	numpy array of x positions
	y		-	numpy array of y positions
	time		-	numpy array of tracker timestamps in milliseconds
	keyword arguments
	missing	-	value to be used for missing data (default = 0.0)
	minlen	-	minimal length of saccades in milliseconds; all detected
				saccades with len(sac) < minlen will be ignored
				(default = 5)
	maxvel	-	velocity threshold in pixels/second (default = 40)
	maxacc	-	acceleration threshold in pixels / second**2
				(default = 340)
	
	returns
	Ssac, Esac
			Ssac	-	list of lists, each containing [starttime]
			Esac	-	list of lists, each containing [starttime, endtime, duration, startx, starty, endx, endy]
	"""
	x, y, time = remove_missing(x, y, time, missing)

	# CONTAINERS
	Ssac = []
	Esac = []

	# INTER-SAMPLE MEASURES
	# the distance between samples is the square root of the sum
	# of the squared horizontal and vertical interdistances
	intdist = (numpy.diff(x)**2 + numpy.diff(y)**2)**0.5
	# get inter-sample times
	inttime = numpy.diff(time)
	# recalculate inter-sample times to seconds
	inttime = inttime / 1000.0
	
	# VELOCITY AND ACCELERATION
	# the velocity between samples is the inter-sample distance
	# divided by the inter-sample time
	vel = intdist / inttime
	# the acceleration is the sample-to-sample difference in
	# eye movement velocity
	acc = numpy.diff(vel)

	# SACCADE START AND END
	t0i = 0
	stop = False
	while not stop:
		# saccade start (t1) is when the velocity or acceleration
		# surpass threshold, saccade end (t2) is when both return
		# under threshold
	
		# detect saccade starts
		sacstarts = numpy.where((vel[1+t0i:] > maxvel).astype(int) + (acc[t0i:] > maxacc).astype(int) >= 1)[0]
		if len(sacstarts) > 0:
			# timestamp for starting position
			t1i = t0i + sacstarts[0] + 1
			if t1i >= len(time)-1:
				t1i = len(time)-2
			t1 = time[t1i]
			
			# add to saccade starts
			Ssac.append([t1])
			
			# detect saccade endings
			sacends = numpy.where((vel[1+t1i:] < maxvel).astype(int) + (acc[t1i:] < maxacc).astype(int) == 2)[0]
			if len(sacends) > 0:
				# timestamp for ending position
				t2i = sacends[0] + 1 + t1i + 2
				if t2i >= len(time):
					t2i = len(time)-1
				t2 = time[t2i]
				dur = t2 - t1

				# ignore saccades that did not last long enough
				if dur >= minlen:
					# add to saccade ends
					Esac.append([t1, t2, dur, x[t1i], y[t1i], x[t2i], y[t2i]])
				else:
					# remove last saccade start on too low duration
					Ssac.pop(-1)

				# update t0i
				t0i = 0 + t2i
			else:
				stop = True
		else:
			stop = True
	
	return Ssac, Esac
def remove_missing(x, y, time, missing):
	mx = numpy.array(x==missing, dtype=int)
	my = numpy.array(y==missing, dtype=int)
	x = x[(mx+my) != 2]
	y = y[(mx+my) != 2]
	time = time[(mx+my) != 2]
	return x, y, time
# Title and file upload
st.title("Detection of Cognitive Impairment using Deep Learning")
file = st.file_uploader("Upload an Excel file", type=["xlsx","csv"])
if st.button("Submit"):
# Check if a file is uploaded
    if file is not None:
        try:
        # Read the Excel file
            if file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                df = pd.read_excel(file, engine="openpyxl")
            elif file.type == "text/csv":
                df=pd.read_csv(file)
            st.write("### File Contents")
            st.write(df)
            root_dir="D:\Downloads\DATA"
            res1=[]
            res2=[]
            res3=[]
            res4=[]
            for i in os.listdir(root_dir):
                path = os.path.join(root_dir,i)
                ps=sorted(os.listdir(path))
                for j in range(4):
                    p=ps[j]
                    a=os.path.join(path,p)
                    c=sorted(list(os.listdir(a)))
                    b=os.path.join(a,c[0])
                    d=os.path.join(a,c[1])
                    e=os.path.join(a,c[2])
                    f=os.path.join(a,c[3])
                    res1.append(b)
                    res2.append(d)
                    res3.append(e)
                    res4.append(f)
            average_left_saccade_duration = []
            average_right_saccade_duration = []
            average_right_pupil_diameter = []
            average_left_pupil_diameter = []
            left_saccade_duration = []
            right_saccade_duration = []
            right_fixation_duration = []
            left_fixation_duration = []
            average_right_fixation_duration = []
            average_left_fixation_duration = []
            fixation_time = []
            fixation_count = []
            saccade_count = []
            left_amplitude=[]
            right_amplitude=[]
            amplitude=[]
            minamp=[]
            maxamp=[]
            left_velocity=[]
            right_velocity=[]
            velocity=[]
            minvel=[]
            maxvel=[]
            left_blink=[]
            right_blink=[]
            blink_duration=[]
            blink_count=[]

            for i in range(len(res1)):
                df = pd.read_csv(res1[i])
                df1 = df.loc[:,["Time","Left Pupil Pos X","Left Pupil Pos Y","Right Pupil Pos X","Right Pupil Pos Y","Right Pupil Diameter (mm)","Left Pupil Diameter (mm)"]]
                data1,data2 = saccade_detection(np.array(df1["Left Pupil Pos X"]),np.array(df1["Left Pupil Pos Y"]) ,np.array(df1["Time"]))
                data3,data4 = saccade_detection(np.array(df["Right Pupil Pos X"]),np.array(df1["Right Pupil Pos Y"]) , np.array(df1["Time"]))
                data5,data6 = fixation_detection(np.array(df1["Left Pupil Pos X"]),np.array(df1["Left Pupil Pos Y"]) ,np.array(df1["Time"]))
                data7,data8 = fixation_detection(np.array(df1["Right Pupil Pos X"]),np.array(df1["Right Pupil Pos Y"]) , np.array(df1["Time"]))
                data9,data10 = blink_detection(np.array(df1["Left Pupil Pos X"]),np.array(df1["Left Pupil Pos Y"]) ,np.array(df1["Time"]))
                data11,data12 = blink_detection(np.array(df1["Right Pupil Pos X"]),np.array(df1["Right Pupil Pos Y"]) , np.array(df1["Time"]))
                for i in range(len(data2)):
                     left_saccade_duration.append(data2[i][2])
                for i in range(len(data4)):
                    right_saccade_duration.append(data4[i][2])
                for i in range(len(data6)):
                    left_fixation_duration.append(data6[i][2])
                for i in range(len(data8)):
                    right_fixation_duration.append(data8[i][2])
                for i in range(len(data2)):
                    left_amplitude.append(((data2[i][3]-data2[i][5])**2+(data2[i][4]-data2[i][6])**2)**0.5)
                for i in range(len(data4)):
                    right_amplitude.append(((data4[i][3]-data4[i][5])**2+(data4[i][4]-data4[i][6])**2)**0.5)
                for (i,j) in zip(left_amplitude,right_amplitude):
                    amplitude.append(i+j)
                for i in range(len(data2)):
                    left_velocity.append((((data2[i][3]-data2[i][5])**2+(data2[i][4]-data2[i][6])**2)**0.5)/data2[i][2])
                for i in range(len(data4)):
                    right_velocity.append((((data4[i][3]-data4[i][5])**2+(data4[i][4]-data4[i][6])**2)**0.5)/data4[i][2])
                for (i,j) in zip(left_velocity,right_velocity):
                    velocity.append(i+j)
                for i in range(len(data10)):
                    left_blink.append(data10[i][2])
                for i in range(len(data12)):
                    right_blink.append(data12[i][2])
                for (i,j) in zip(left_blink,right_blink):
                    blink_duration.append(i+j)
                fixation_time.append((df1.iloc[-1,0] - df1.iloc[0,0])/1000)
                saccade_count.append(len(data2))
                fixation_count.append(len(data6))
                blink_count.append(len(data10))
                average_left_saccade_duration.append(mean(left_saccade_duration))
                average_right_saccade_duration.append(mean(right_saccade_duration))
                average_left_fixation_duration.append(mean(left_fixation_duration))
                average_right_fixation_duration.append(mean(right_fixation_duration))
                average_right_pupil_diameter.append(mean(df1.iloc[:,5]))
                average_left_pupil_diameter.append(mean(df1.iloc[:,6]))
                minamp.append(min(amplitude))
                maxamp.append(max(amplitude))
                minvel.append(min(velocity))
                maxvel.append(max(velocity))
                TMT_A1 = pd.DataFrame(list(zip(average_left_saccade_duration, average_right_saccade_duration,average_right_pupil_diameter,average_left_pupil_diameter,average_left_fixation_duration,average_right_fixation_duration,fixation_time,saccade_count,fixation_count,left_amplitude,right_amplitude,amplitude,minamp,maxamp,left_velocity,right_velocity,velocity,minvel,maxvel,left_blink,right_blink,blink_duration,blink_count)),columns =["average_left_saccade_duration","average_right_saccade_duration","average_right_pupil_diameter","average_left_pupil_diameter","average_left_fixation_duration","average_right_fixation_duration","fixation_time","saccade_count","fixation_count","left_saccade_amplitude","right_saccade_amplitude","saccade_amplitude","minimum_amplitude","maximum_amplitude","left_velocity","right_velocity","saccade_velocity","minimum_velocity","maximum_velocity","left_blink","right_blink","blink_duration","blink_count"])
            TMT_A1['mmse_score']=[random.randint(21,30) for i in range(len(TMT_A1))]
            #TMT_A1=pd.read_excel("D:\Downloads\TMT_A12.xlsx")
            TMT_A1["label"] = "None" *len(TMT_A1["mmse_score"])
            for i in range(len(TMT_A1["mmse_score"])):
                if 20 < TMT_A1["mmse_score"][i] <= 25:
                    TMT_A1["label"][i] = 0
                else:
                    TMT_A1["label"][i] = 1
            X = TMT_A1.drop(['label','mmse_score'],axis=1)
            y = TMT_A1[["label"]]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            X_train = np.asarray(X_train).astype(np.float32)
            y_train = np.asarray(y_train).astype(np.float32)
            x_test = np.asarray(X_test).astype(np.float32)
            y_test = np.asarray(y_test).astype(np.float32)
            X_val = np.asarray(X_val).astype(np.float32)
            y_val = np.asarray(y_val).astype(np.float32)
            model = Sequential()
            model.add(Dense(64, input_dim=23, activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))

        # compile the model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # define the early stopping criterion
            early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

        # fit the model with early stopping
            history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

        # evaluate the model on the test set
            loss, accuracy = model.evaluate(X_test, y_test)

            y_pred = model.predict(X_test)
            y_pred = (y_pred > 0.5).astype(int)
            if(y_pred[0]==0):
                result="Have Cognitive Impairment and effected with dimentia"
            else:
                result="Not effected with dimentia"
            st.success(result)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please upload an Excel file before clicking the Submit button.")
