from sklearn.preprocessing import MinMaxScaler
import numpy as np
import keras.models as models
import keras.layers as layers
import keras.optimizers as optimizers
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Convolution2D, ZeroPadding2D, AveragePooling2D
import keras.backend as K
import joblib
import base64
import cv2
from keras.layers.advanced_activations import PReLU

from flask import Flask, request, jsonify
app = Flask(__name__)
R = np.matrix(np.zeros([6, 6]))
Q = np.matrix(np.zeros([6, 6]))

armsweek = 0
legsweek = 0
absweek = 0

fitnesslevel = 0

scaler_diabetic_complications = MinMaxScaler()
scaler_diabetic_risk = MinMaxScaler()

data_diabetic_complications = np.load('data_diabetic_complications.npy')
data_diabetic_risk = np.load('data_diabetic_risk.npy')

scaler_diabetic_complications.fit(data_diabetic_complications)
scaler_diabetic_risk.fit(data_diabetic_risk)


def load_stress_model():
    K.clear_session()
    img_rows, img_cols = 48, 48
    model = models.Sequential()

    model.add(Convolution2D(64, 5, 5, border_mode='valid',
                            input_shape=(img_rows, img_cols, 1)))
    model.add(PReLU(init='zero', weights=None))
    model.add(ZeroPadding2D(padding=(2, 2), dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
    model.add(Convolution2D(64, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
    model.add(Convolution2D(64, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
    model.add(Convolution2D(128, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
    model.add(Convolution2D(128, 3, 3))
    model.add(PReLU(init='zero', weights=None))

    model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(PReLU(init='zero', weights=None))
    model.add(Dropout(0.2))
    model.add(Dense(1024))
    model.add(PReLU(init='zero', weights=None))
    model.add(Dropout(0.2))

    model.add(Dense(6))

    model.add(Activation('softmax'))

    ada = optimizers.Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=ada, metrics=['accuracy'])

    model.load_weights('emotion_recognition.h5')

    return model


def food_model():
    K.clear_session()
    model = models.Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=(50, 50, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Dense(8))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.load_weights('Food_V1.h5')
    return model

def load_model(weight_file, input_size):

    K.clear_session()

    model = models.Sequential()
    model.add(layers.Dense(128, input_dim=input_size,
                           kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.load_weights(weight_file)
    # model.summary()
    return model


@app.route('/diabetic_complications', methods=['POST', 'GET'])
def diabetic_complications():

    User_json = request.json

    sex = int(User_json['sex'])
    age = int(User_json['age'])
    bmi = float(User_json['bmi'])
    dia_duration = int(User_json['dia_duration'])
    insulin = int(User_json['insulin'])
    medi_treatment = int(User_json['medi_treatment'])
    hba1c = float(User_json['hba1c'])

    test_data = [sex, age, bmi, dia_duration, insulin, medi_treatment, hba1c]

    print(test_data)

    test_data = scaler_diabetic_complications.transform([test_data])

    model_nephropathy = load_model('P1_compliactions_nephropathy.h5', 7)
    result_nephropathy = round(
        model_nephropathy.predict(test_data)[0][0]*100, 2)

    model_retinopathy = load_model('P1_compliactions_retinopathy.h5', 7)
    result_retinopathy = round(
        model_retinopathy.predict(test_data)[0][0]*100, 2)

    model_neuropathy = load_model('P1_compliactions_neuropathy.h5', 7)
    result_neuropathy = round(model_neuropathy.predict(test_data)[0][0]*100, 2)

    model_foot_ulcer = load_model('P1_compliactions_foot_ulcer.h5', 7)
    result_foot_ulcer = round(model_foot_ulcer.predict(test_data)[0][0]*100, 2)

    print(result_nephropathy, result_retinopathy,
          result_neuropathy, result_foot_ulcer)

    results = [{'result_nephropathy': result_nephropathy, 'result_retinopathy': result_retinopathy,
                'result_neuropathy': result_neuropathy, 'result_foot_ulcer': result_foot_ulcer}]

    return jsonify(results=results)


@app.route('/diabetic_risk', methods=['POST', 'GET'])
def diabetic_risk():

    User_json = request.json

    print("model hit")
    print(User_json)

    age = int(User_json['age'])
    sex = int(User_json['sex'])
    height = float(User_json['height'])
    weight = float(User_json['weight'])
    sbp = float(User_json['sbp'])
    dbp = float(User_json['dbp'])
    chol = float(User_json['chol'])
    trigl = float(User_json['trigl'])
    hdl = float(User_json['hdl'])
    ldl = float(User_json['ldl'])
    smoke = int(User_json['smoke'])
    drink = float(User_json['drink'])
    family = float(User_json['family'])

    test_data = [age, sex, height, weight, sbp, dbp, chol,
                 trigl, hdl, ldl,smoke, drink, family]

    test_data = scaler_diabetic_risk.transform([test_data])

    print(test_data)

    model = load_model('P1_Diabetic_Risk.h5', 13)
    result = round(model.predict(test_data)[0][0]*100, 2)

    if(result>1 and result<10):
        result=result/10*100

    print(result)

    results = [{'result': result}]

    return jsonify(results=results)


@app.route('/')
def index():

    return "<h1>HELLO HEROKU!</h1>"


@app.route('/meal_plan', methods=['POST', 'GET'])
def meal_plan():

    User_json = request.json

    sex = int(User_json['sex'])
    age = int(User_json['age'])
    bmi = float(User_json['bmi'])
    risk = float(User_json['risk'])

    test_data = np.array([sex, age, bmi, risk]).reshape(1, -1)

    model = joblib.load('2_meal_plan.sav')
    print(model.predict(test_data))
    result = model.predict(test_data)[0]
    results = [{'result': int(result)}]
    return jsonify(results=results)


@app.route('/food', methods=['POST', 'GET'])
def food():

    print('came here')
    model = food_model()

    print('after model')

    target_dict = {0: 'Burger', 1: 'Hoppers', 2: 'Noodles', 3: 'Pizza', 4: 'Rice', 5: 'Rolls', 6: 'Rottie', 7: 'Samosa'}

    User_json = request.json

    encrypted_string = User_json['base_string']

    imgdata = base64.b64decode(encrypted_string)
    filename = 'test_image.jpg'
    with open(filename, 'wb') as f:
        f.write(imgdata)

    img = cv2.imread('test_image.jpg')
    test_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (height, width) = img.shape[:2]

    #imgdata[0:50,0:width] = [0,0,255]
    #imgdata[50:80,0:100] = [0,255,255]

    test_img = cv2.resize(test_img, (50, 50))
    test_img = test_img/255.0
    test_img = test_img.reshape(-1, 50, 50, 1)
    result = model.predict([test_img])


    label = target_dict[np.argmax(result)]

    results = [
        {
            "predictedResult": label
        }
    ]
    return jsonify(results=results)


@app.route('/activity_suggestion', methods=['POST', 'GET'])
def activity_suggestion():

    User_json = request.json

    sex = int(User_json['sex'])
    age = int(User_json['age'])
    employ = float(User_json['employ'])
    workHours = float(User_json['workHours'])
    freeHours = float(User_json['freeHours'])
    sleepHours = float(User_json['sleepHours'])
    famHours = float(User_json['famHours'])
    ill = float(User_json['ill'])
    emotion = float(User_json['emotion'])
    score = float(User_json['score'])

    test_data = np.array([sex, age, employ, workHours, freeHours,
                          sleepHours, famHours, ill, emotion, score]).reshape(1, -1)

    model = joblib.load('4_activity_suggestion.sav')
    print(model.predict(test_data))
    result = model.predict(test_data)[0]
    results = [{'result': int(result)}]
    return jsonify(results=results)


@app.route('/excercise_plan', methods=['POST', 'GET'])
def excercise_plan():

    User_json = request.json

    sex = int(User_json['sex'])
    age = int(User_json['age'])
    bmi = float(User_json['bmi'])
    risk = float(User_json['risk'])

    test_data = np.array([sex, age, bmi, risk]).reshape(1, -1)

    model = joblib.load('4_excersices.sav')
    print(model.predict(test_data))
    result = model.predict(test_data)[0]
    results = [{'result': int(result)}]
    return jsonify(results=results)


@app.route('/stress', methods=['POST', 'GET'])
def stress():
	model = load_stress_model()
	emotions = {0: 'Angry', 1: 'Fear', 2: 'Happy',3: 'Sad', 4: 'Surprised', 5: 'Neutral'}
	User_json = request.json
	encrypted_string = User_json['base_string']
	imgdata = base64.b64decode(encrypted_string)
	filename = 'emotion_image.jpg'
	with open(filename, 'wb') as f:
		f.write(imgdata)

	faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	frame = cv2.imread('emotion_image.jpg')
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(100, 100))
	(height, width) = frame.shape[:2]
	frame[0:50, 0:230] = [100]
	frame[0:50, 230:width] = [100]
	frame[50:80, 0:100] = [50]

	try:
		if (len(faces)) > 0:
			for x, y, width, height in faces:
				cropped_face = gray[y:y + height, x:x + width]
				test_image = cv2.resize(cropped_face, (48, 48))
				test_image = test_image.reshape(-1, 48, 48, 1)
				test_image = np.multiply(test_image, 1.0 / 255.0)
				probab = model.predict(test_image)[0]
				label = np.argmax(probab)
				probab_predicted = int(probab[label])
				predicted_emotion = emotions[label]
				print(predicted_emotion)
				acc = np.max(probab)
				cv2.rectangle(frame, (x, y), (x+width, y+height),(0, 255, 0), 2)
				cv2.putText(frame, predicted_emotion, (10, 35),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
				cv2.putText(frame, 'acc:'+str(round(acc, 2)), (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
				cv2.imwrite('emotion_result.jpg', frame)

				results = [{"predictedResult": predicted_emotion}]

				return jsonify(results=results)

	except Exception as e:

		results = [{"predictedResult": 0}]
		print(e)
		return jsonify(results=results)

	results = [{"predictedResult": 0}]
	return jsonify(results=results)
#######################################################################################


# app.run(debug=True)

# Get the Nearest values from the provided array


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# This function returns all available actions in the state given as an argument


def available_actions(state):
    # getting the row in the R matrix which the state represent
    current_state_row = R[state, ]
    # getting the row indexs of the R matrix which are not -1
    av_act = np.where(current_state_row >= 0)[1]
    return av_act

# This function chooses at random which action to be performed within the range
# of all the availalbe actions


def sample_next_actions(available_act):
    # choosing the next action randomly from availalbe actions
    next_action = int(np.random.choice(available_act, 1))
    return next_action

# This function updates the Q matrix according to the path selected and the Q
# learning algorithm


def update(current_state, action, gamma):
    # checking what is the index which holds the maximum amount
    max_index = np.where(Q[action, ] == np.max(Q[action, ]))[1]

    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size=1))
    else:
        max_index = int(max_index)
    max_value = Q[action, max_index]

    # Q learning formula
    Q[current_state, action] = R[current_state, action] + gamma + max_value

# Run Q Learning algorithm to get the excersice schedule


def getexcericelist(fitnesevaluation):

    if (fitnesevaluation < 4):
        R = np.matrix([[-1, 0, 0, 0, 0, -1],
                       [0, -1, 0, 0, 0, -1],
                       [0, 0, -1, 0, 0, -1],
                       [0, 0, 0, -1, 0, -1],
                       [0, 0, 0, 0, -1, 100],
                       [0, 0, 0, 0, 0, -1]])
    elif (fitnesevaluation < 7):
        R = np.matrix([[-1, 0, 0, 0, 0, 100],
                       [0, -1, 0, 0, 0, 100],
                       [0, 0, -1, 0, 0, 100],
                       [0, 0, 0, -1, 0, -1],
                       [0, 0, 0, 0, -1, -1],
                       [0, 0, 0, 0, 0, -1]])
    else:
        R = np.matrix([[-1, 0, 0, 0, 0, 100],
                       [0, -1, 0, 0, 0, 100],
                       [0, 0, -1, 0, 0, 100],
                       [0, 0, 0, -1, 0, 100],
                       [0, 0, 0, 0, -1, 100],
                       [0, 0, 0, 0, 0, -1]])

    # Q matrix
    Q = np.matrix(np.zeros([6, 6]))

    # Gamma (learning parameter)
    gamma = 0.8

    # Initial state. (Usually to be chosen at random)
    initial_state = 1

    # Get available actions in the current state
    available_act = available_actions(initial_state)

    # Sample next action to be performed
    action = sample_next_actions(available_act)

    # Update Q matrix
    update(initial_state, action, gamma)

    # -----------------------------------------------------------------------------
    # Training
    for i in range(10000):
        current_state = np.random.randint(0, int(Q.shape[0]))
        available_act = available_actions(current_state)
        action = sample_next_actions(available_act)
        update(current_state, action, gamma)

    # Normalize the "trained" Q matrix
    print(Q / np.max(Q) * 100)

    # -------------------------------------------------------------------------------
    # Testing

    current_state = 0
    steps = [current_state]

    infinitloopstop = 1
    while current_state != 5:

        infinitloopstop = infinitloopstop + 1

        next_step_index = np.where(
            Q[current_state, ] == np.max(Q[current_state, ]))[1]

        if next_step_index.shape[0] > 1:
            next_step_index = int(np.random.choice(next_step_index, size=1))
        else:
            next_step_index = int(next_step_index)

        steps.append(next_step_index)
        current_state = next_step_index

        if (infinitloopstop > 1000):
            break

    # print selected sequence of steps
    steps = list(dict.fromkeys(steps))
    return steps

# User fitness evaluation for running


def legsfitnessevaluation(user_age, user_gender, user_spenttime, age_array, womentime_array, mentime_array):

    timeindex = age_array.index(find_nearest(age_array, user_age))

    if (user_gender == 'male'):
        timearray = mentime_array
    else:
        timearray = womentime_array

    if (timearray[timeindex] >= user_spenttime):
        fitnesevaluation = 10
    elif (timearray[timeindex] * 2 <= user_spenttime):
        fitnesevaluation = 1
    else:
        fitnesevaluation = ((timearray[timeindex] - 2 * 2) / 10) - (
            user_spenttime - timearray[timeindex])
        fitnesevaluation = 10 - abs(fitnesevaluation)

    return fitnesevaluation

# User fitness evaluation for arms


def armsfitnessevaluation(user_age, user_gender, activitymeasurement, age_array, womentime_array, mentime_array):

    timeindex = age_array.index(find_nearest(age_array, user_age))

    if (user_gender == 'male'):
        timearray = mentime_array
    else:
        timearray = womentime_array

    if (timearray[timeindex] <= activitymeasurement):
        fitnesevaluation = 10
    else:
        fitnesevaluation = float(10) / timearray[timeindex]
        fitnesevaluation = fitnesevaluation * activitymeasurement

    return fitnesevaluation

# User fitness evaluation for abs


def absfitnessevaluation(user_age, activitymeasurement, age_array, hearratemin_array, hearratemax_array):

    timeindex = age_array.index(find_nearest(age_array, user_age))

    if(hearratemin_array[timeindex] > activitymeasurement):
        fitnesevaluation = 1
    elif(hearratemax_array[timeindex] < activitymeasurement):
        fitnesevaluation = 10
    else:
        fitnesevaluation = float(
            10) / (hearratemax_array[timeindex] - hearratemin_array[timeindex])
        fitnesevaluation = fitnesevaluation * \
            (activitymeasurement - hearratemin_array[timeindex])

    return fitnesevaluation

# Get the excerise list based on the Q learning outcome


def excersielistforlegs(user_age, user_gender, activitymeasurement):

    age_array = [25, 35, 45, 55, 65]
    womentime_array = [13, 13.5, 14, 16, 17.5]
    mentime_array = [11, 11.5, 12, 13, 14]

    global fitnesslevel
    fitnesslevel = legsfitnessevaluation(
        user_age, user_gender, activitymeasurement, age_array, womentime_array, mentime_array)

    excersiselist = []
    excersiselist = getexcericelist(fitnesslevel)

    excersiselistbeginner = [
        {"exKey": 1, "name": "Side Hop","cal":0.8,
            "imageUrl": "https://cdn-ami-drupal.heartyhosting.com/sites/muscleandfitness.com/files/_main2_sidetosidehop.jpg", "level": "beginner"},
        {"exKey": 2, "name": "Squats", "cal":11.6,"imageUrl": "https://media1.popsugar-assets.com/files/thumbor/_8L5Z28n57dzzdPT8pEV2Gmup50/fit-in/1024x1024/filters:format_auto-!!-:strip_icc-!!-/2016/03/04/675/n/1922398/2a4b0a04f46626f9_squat.jpg", "level": "beginner"},
        {"exKey": 3, "name": "Backward Lunge","cal":3.44,
            "imageUrl": "https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/goblet-reverse-lunge-1441211036.jpg", "level": "beginner"},
        {"exKey": 4, "name": "Wall Calf Raises","cal":1.28,
            "imageUrl": "https://www.aliveandwellfitness.ca/wp-content/uploads/2015/02/Calf-Raise.jpg", "level": "beginner"},
        {"exKey": 5, "name": "Sumo Squat Calf Raises with Wall","cal":4.96,
            "imageUrl": "https://static.onecms.io/wp-content/uploads/sites/35/2012/06/16185952/plie-squat-calf-420_0.jpg", "level": "beginner"},
        {"exKey": 6, "name": "Knee to Chest", "cal":20,"imageUrl": "https://www.verywellhealth.com/thmb/EpgYtdLIAW9nSX0pWozlcvNJSiY=/3525x2350/filters:no_upscale():max_bytes(150000):strip_icc()/Depositphotos_22103221_original-56a05fe85f9b58eba4b027a7.jpg", "level": "beginner"}
    ]

    excersiselistintermediate = [
        {"exKey": 7,  "name": "Jumping Jacks", "cal":64,"imageUrl": "https://media1.popsugar-assets.com/files/thumbor/BLn2-1T1Yp-cgpoOU76QVkuhlpc/fit-in/2048xorig/filters:format_auto-!!-:strip_icc-!!-/2015/05/01/974/n/1922729/8a48e47672d474dc_c9d6640d1d97a449_jumping-jacks.xxxlarge/i/Jumping-Jacks.jpg", "level": "medium"},
        {"exKey": 8,  "name": "Backward Lunge","cal":3.44,
            "imageUrl": "https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/goblet-reverse-lunge-1441211036.jpg", "level": "medium"},
        {"exKey": 9,  "name": "Wall Calf Raises","cal":1.28,
            "imageUrl": "https://www.aliveandwellfitness.ca/wp-content/uploads/2015/02/Calf-Raise.jpg", "level": "medium"},
        {"exKey": 10, "name": "Calf Raise with Splayed Foot","cal":2.4,
            "imageUrl": "https://i.ytimg.com/vi/-M4-G8p8fmc/maxresdefault.jpg", "level": "medium"},
        {"exKey": 11, "name": "Wall Sit","cal":48, "imageUrl": "http://s3.amazonaws.com/prod.skimble/assets/4655/skimble-workout-trainer-exercise-wall-sit-tip-toes-1_iphone.jpg", "level": "medium"},
        {"exKey": 54, "name": "step up","cal":1.28,
            "imageUrl": "https://www.mensjournal.com/wp-content/uploads/mf/db-stepup-the-30-best-legs-exercises-of-all-time.jpg", "level": "medium"}
    ]

    excersiselistadvanced = [
        {"exKey": 12, "name": "Burpees","cal":4,
            "imageUrl": "https://cdn4.vectorstock.com/i/1000x1000/82/93/burpees-exercise-vector-25048293.jpg", "level": "hard"},
        {"exKey": 13, "name": "Squats", "cal":11.6,"imageUrl": "https://media1.popsugar-assets.com/files/thumbor/_8L5Z28n57dzzdPT8pEV2Gmup50/fit-in/1024x1024/filters:format_auto-!!-:strip_icc-!!-/2016/03/04/675/n/1922398/2a4b0a04f46626f9_squat.jpg", "level": "hard"},
        {"exKey": 14, "name": "Curtsy Lunge","cal":6, "imageUrl": "https://media1.popsugar-assets.com/files/thumbor/CAYVOgmZ__WZZpt1ReKTUOaSsY4/fit-in/1024x1024/filters:format_auto-!!-:strip_icc-!!-/2015/12/16/664/n/1922398/19ccad6a3187b053_Side-Lunge-Curtsy-Squat.jpg", "level": "hard"},
        {"exKey": 15, "name": "Jumpying Squats","cal":4.96, "imageUrl": "https://media1.popsugar-assets.com/files/thumbor/_gsXN6w15Fm3hLGdCX-rRUAv5vs/fit-in/1024x1024/filters:format_auto-!!-:strip_icc-!!-/2014/01/31/901/n/1922729/1545977b1743e558_Jump-Squat.jpg", "level": "hard"},
        {"exKey": 16, "name": "Lying with Butterfly Stretch","cal":24,
            "imageUrl": "http://s3.amazonaws.com/prod.skimble/assets/1259100/image_iphone.jpg", "level": "hard"},
        {"exKey": 17, "name": "Wall Sit","cal":48,
            "imageUrl": "http://s3.amazonaws.com/prod.skimble/assets/4655/skimble-workout-trainer-exercise-wall-sit-tip-toes-1_iphone.jpg", "level": "hard"}
    ]

    excericesheudle = []

    for x in excersiselist:
        if(fitnesslevel < 4):
            excericesheudle.append(excersiselistbeginner[x])
        elif(fitnesslevel < 7):
            excericesheudle.append(excersiselistintermediate[x])
        else:
            excericesheudle.append(excersiselistadvanced[x])

    return excericesheudle


def excersielistforarms(user_age, user_gender, activitymeasurement):

    age_array = [25, 35, 45, 55, 65]
    womentime_array = [20, 19, 14, 10, 10]
    mentime_array = [28, 21, 16, 12, 11]

    global fitnesslevel
    fitnesslevel = armsfitnessevaluation(user_age, user_gender, activitymeasurement, age_array, womentime_array,
                                         mentime_array)

    excersiselist = []
    excersiselist = getexcericelist(fitnesslevel)

    excersiselistbeginner = [
        {"exKey": 18, "name": "Arm Raises","cal":3.6,
            "imageUrl": "https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/0504-front-arm-raise-1441032989.jpg", "level": "beginner"},
        {"exKey": 19, "name": "Side Arm Raise","cal":1.56,
            "imageUrl": "https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/1205-armraise-1441032989.jpg", "level": "beginner"},
        {"exKey": 20, "name": "Triceps Dips","cal":0.8,
            "imageUrl": "https://wiki-fitness.com/wp-content/uploads/2014/04/triceps-dips.jpg", "level": "beginner"},
        {"exKey": 21, "name": "Arm Circles Clockwise","cal":3.24,
            "imageUrl": "http://www.backatsquarezero.com/wp-content/uploads/2017/10/Front-Raise-Circles.jpg", "level": "beginner"},
        {"exKey": 22, "name": "Arm Circles CounterClockWise","cal":9.36,
            "imageUrl": "http://www.igophysio.co.uk/wp-content/uploads/2017/03/arm-circles.jpg", "level": "beginner"},
        {"exKey": 23, "name": "Diamond Push-Ups","cal":84,
            "imageUrl": "https://s3.amazonaws.com/prod.skimble/assets/4141/skimble-workout-trainer-exercise-kneeling-diamond-push-ups-2_iphone.jpg", "level": "beginner"}
    ]

    excersiselistintermediate = [
        {"exKey": 24, "name": "Arm Raises","cal":3.6,
            "imageUrl": "https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/0504-front-arm-raise-1441032989.jpg", "level": "medium"},
        {"exKey": 25, "name": "Side Arm Raise","cal":1.56,
            "imageUrl": "https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/1205-armraise-1441032989.jpg", "level": "medium"},
        {"exKey": 26, "name": "Floor Tricep Dips","cal":1.92, "imageUrl": "https://media1.popsugar-assets.com/files/thumbor/1Dp0qN0aVQd9lEApVaq93VAoG3k/fit-in/2048xorig/filters:format_auto-!!-:strip_icc-!!-/2014/04/25/773/n/1922729/71dfb70a7ef96be8_c58b14640c2032c6_triceps-dips.jpg.xxxlarge/i/Triceps-Dips.jpg", "level": "medium"},
        {"exKey": 27, "name": "Military Push Ups","cal":3.72,
            "imageUrl": "https://bodylastics.com/wp-content/uploads/2018/08/resisted-decline-military-push-up.jpg", "level": "medium"},
        {"exKey": 28, "name": "Alternative Hooks","cal":3,
            "imageUrl": "https://redefiningstrength.com/wp-content/uploads/2016/04/lunge-and-reach-e1461966612120.jpg", "level": "medium"},
        {"exKey": 29, "name": "Push-up & Rotation", "cal":42,"imageUrl": "https://media1.popsugar-assets.com/files/thumbor/24DvTMytVexDVCjeWQi-IqjI8M8/fit-in/2048xorig/filters:format_auto-!!-:strip_icc-!!-/2014/09/19/009/n/1922729/76aca72c86654a36_11-Push-Up-Rotation/i/Push-Up-Rotation.jpg", "level": "medium"}
    ]

    excersiselistadvanced = [
        {"exKey": 30, "name": "Arm Raises","cal":3.6,
            "imageUrl": "https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/0504-front-arm-raise-1441032989.jpg", "level": "hard"},
        {"exKey": 31, "name": "Side Arm Raise","cal":1.56,
            "imageUrl": "https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/1205-armraise-1441032989.jpg", "level": "hard"},
        {"exKey": 32, "name": "Skipping WithOut Rope","cal":1.92,
            "imageUrl": "https://lifeinleggings.com/wp-content/uploads/2016/03/invisible-jump-rope-exercise.jpg", "level": "hard"},
        {"exKey": 33, "name": "Burpees","cal":6,
            "imageUrl": "https://cdn4.vectorstock.com/i/1000x1000/82/93/burpees-exercise-vector-25048293.jpg", "level": "hard"},
        {"exKey": 34, "name": "Floor Tricep Dips","cal":1.92, "imageUrl": "https://media1.popsugar-assets.com/files/thumbor/1Dp0qN0aVQd9lEApVaq93VAoG3k/fit-in/2048xorig/filters:format_auto-!!-:strip_icc-!!-/2014/04/25/773/n/1922729/71dfb70a7ef96be8_c58b14640c2032c6_triceps-dips.jpg.xxxlarge/i/Triceps-Dips.jpg", "level": "hard"},
        {"exKey": 35, "name": "Military Push Ups","cal":3.72,
            "imageUrl": "https://bodylastics.com/wp-content/uploads/2018/08/resisted-decline-military-push-up.jpg", "level": "hard"}
    ]

    excericesheudle = []

    for x in excersiselist:
        if (fitnesslevel < 4):
            excericesheudle.append(excersiselistbeginner[x])
        elif (fitnesslevel < 7):
            excericesheudle.append(excersiselistintermediate[x])
        else:
            excericesheudle.append(excersiselistadvanced[x])

    return excericesheudle


def excersielistforabs(user_age, activitymeasurement):

    age_array = [25, 35, 45, 55, 65]
    hearratemin_array = [98, 93, 88, 83, 78]
    hearratemax_array = [146, 138, 131, 123, 116]

    global fitnesslevel
    fitnesslevel = absfitnessevaluation(user_age, activitymeasurement, age_array, hearratemin_array,
                                        hearratemax_array)

    excersiselist = []
    excersiselist = getexcericelist(fitnesslevel)

    excersiselistbeginner = [
        {"exKey": 36, "name": "Abdominal Crunches","cal":1.272,
            "imageUrl": "https://3i133rqau023qjc1k3txdvr1-wpengine.netdna-ssl.com/wp-content/uploads/2014/07/Basic-Crunch_Exercise.jpg", "level": "beginner"},
        {"exKey": 37, "name": "Russian Twist", "cal":3.28,"imageUrl": "https://media1.popsugar-assets.com/files/thumbor/Sy9BdUS395G5YSZL2ErQcNxPBSI/fit-in/1024x1024/filters:format_auto-!!-:strip_icc-!!-/2016/09/14/943/n/1922729/75a989e2_Core-Seated-Russian-Twist/i/Circuit-3-Exercise-4-Seated-Russian-Twist.jpg", "level": "beginner"},
        {"exKey": 38, "name": "Mountain Climber","cal":0.8,
            "imageUrl": "https://rejuvage.com/wp-content/uploads/2019/07/iStock-957699448.jpg", "level": "beginner"},
        {"exKey": 39, "name": "Heel Touch","cal":2.4,
            "imageUrl": "https://i.pinimg.com/originals/96/ae/f4/96aef451dc1b91511d810b33b0c595ff.jpg", "level": "beginner"},
        {"exKey": 40, "name": "Leg Raises","cal":3.28,
            "imageUrl": "https://i.ytimg.com/vi/Wp4BlxcFTkE/maxresdefault.jpg", "level": "beginner"},
        {"exKey": 41, "name": "Cobra Stretch","cal":8,
            "imageUrl": "http://media4.onsugar.com/files/2013/07/17/920/n/1922729/699256f565453684_kristin.jpg", "level": "beginner"}
    ]

    excersiselistintermediate = [
        {"exKey": 42, "name": "Crossover Crunch","cal":1.908,
            "imageUrl": "https://i.ytimg.com/vi/C4MbUFxLm2Y/hqdefault.jpg", "level": "medium"},
        {"exKey": 43, "name": "Mountain Climber","cal":1.2,
            "imageUrl": "https://rejuvage.com/wp-content/uploads/2019/07/iStock-957699448.jpg", "level": "medium"},
        {"exKey": 44, "name": "Bicycle Crunches","cal":1.908,
            "imageUrl": "https://bodylastics.com/wp-content/uploads/2018/08/Bicycle-Abs-Crunches.jpg", "level": "medium"},
        {"exKey": 45, "name": "Heel Touch","cal":3.6,
            "imageUrl": "https://i.pinimg.com/originals/96/ae/f4/96aef451dc1b91511d810b33b0c595ff.jpg", "level": "medium"},
        {"exKey": 46, "name": "Leg Raises","cal":4.92,
            "imageUrl": "https://i.ytimg.com/vi/Wp4BlxcFTkE/maxresdefault.jpg", "level": "medium"},
        {"exKey": 47, "name": "V-Up","cal":3.6,
            "imageUrl": "https://gethealthyu.com/wp-content/uploads/2014/09/V-Up_Exercise-2.jpg", "level": "medium"}
    ]

    excersiselistadvanced = [
        {"exKey": 48, "name": "Push-Up & Rotation","cal":42, "imageUrl": "https://media1.popsugar-assets.com/files/thumbor/24DvTMytVexDVCjeWQi-IqjI8M8/fit-in/2048xorig/filters:format_auto-!!-:strip_icc-!!-/2014/09/19/009/n/1922729/76aca72c86654a36_11-Push-Up-Rotation/i/Push-Up-Rotation.jpg", "level": "hard"},
        {"exKey": 49, "name": "Russian Twist","cal":4.92,
            "imageUrl": "https://i.pinimg.com/originals/70/c2/65/70c2652a92ba946bc6a37f13563fda03.jpg", "level": "hard"},
        {"exKey": 50, "name": "Bicycle Crunches","cal":1.908,
            "imageUrl": "https://bodylastics.com/wp-content/uploads/2018/08/Bicycle-Abs-Crunches.jpg", "level": "hard"},
        {"exKey": 51, "name": "Heel Touch","cal":3.6,
            "imageUrl": "https://i.pinimg.com/originals/96/ae/f4/96aef451dc1b91511d810b33b0c595ff.jpg", "level": "hard"},
        {"exKey": 52, "name": "Leg Raises","cal":4.92,
            "imageUrl": "https://i.ytimg.com/vi/Wp4BlxcFTkE/maxresdefault.jpg", "level": "hard"},
        {"exKey": 53, "name": "V-Up","cal":3.6,
            "imageUrl": "https://gethealthyu.com/wp-content/uploads/2014/09/V-Up_Exercise-2.jpg", "level": "hard"}
    ]

    excericesheudle = []

    for x in excersiselist:
        if (fitnesslevel < 4):
            excericesheudle.append(excersiselistbeginner[x])
        elif (fitnesslevel < 7):
            excericesheudle.append(excersiselistintermediate[x])
        else:
            excericesheudle.append(excersiselistadvanced[x])

    return excericesheudle


@app.route('/getlegsshedule', methods=['POST', 'GET'])
def generatesheduleforlegs():

    User_json = request.json

    global legsweek
    legsweek += 1

    global fitnesslevel

    user_age = int(User_json['userage'])
    user_gender = User_json['usergender']
    activitymeasurement = float(User_json['activitymeasurement'])

    weekshedulerlegs = {
        "Monday": excersielistforlegs(user_age, user_gender, activitymeasurement),
        "Tuesday": excersielistforlegs(user_age, user_gender, activitymeasurement),
        "Wednesday": excersielistforlegs(user_age, user_gender, activitymeasurement),
        "Thursday": excersielistforlegs(user_age, user_gender, activitymeasurement),
        "Friday": excersielistforlegs(user_age, user_gender, activitymeasurement),
        "Saturday": excersielistforlegs(user_age, user_gender, activitymeasurement),
        "Sunday": excersielistforlegs(user_age, user_gender, activitymeasurement),
        "WeekNumber": legsweek,
        "fitnessValue": fitnesslevel
    }
    return jsonify(results=weekshedulerlegs)


@app.route('/getarmsshedule', methods=['POST', 'GET'])
def generatesheduleforarms():

    User_json = request.json

    global armsweek
    armsweek += 1

    global fitnesslevel

    user_age = int(User_json['userage'])
    user_gender = User_json['usergender']
    activitymeasurement = int(User_json['activitymeasurement'])

    weekshedulerlegs = {
        "Monday": excersielistforarms(user_age, user_gender, activitymeasurement),
        "Tuesday": excersielistforarms(user_age, user_gender, activitymeasurement),
        "Wednesday": excersielistforarms(user_age, user_gender, activitymeasurement),
        "Thursday": excersielistforarms(user_age, user_gender, activitymeasurement),
        "Friday": excersielistforarms(user_age, user_gender, activitymeasurement),
        "Saturday": excersielistforarms(user_age, user_gender, activitymeasurement),
        "Sunday": excersielistforarms(user_age, user_gender, activitymeasurement),
        "WeekNumber": armsweek,
        "fitnessValue": fitnesslevel
    }
    return jsonify(results=weekshedulerlegs)


@app.route('/getabsshedule', methods=['POST', 'GET'])
def generatesheduleforabs():

    User_json = request.json

    global absweek
    absweek += 1

    global fitnesslevel

    user_age = int(User_json['userage'])
    activitymeasurement = float(User_json['activitymeasurement'])

    weekshedulerlegs = {
        "Monday": excersielistforabs(user_age, activitymeasurement),
        "Tuesday": excersielistforabs(user_age, activitymeasurement),
        "Wednesday": excersielistforabs(user_age, activitymeasurement),
        "Thursday": excersielistforabs(user_age, activitymeasurement),
        "Friday": excersielistforabs(user_age, activitymeasurement),
        "Saturday": excersielistforabs(user_age, activitymeasurement),
        "Sunday": excersielistforabs(user_age, activitymeasurement),
        "WeekNumber": absweek,
        "fitnessValue": fitnesslevel
    }

    return jsonify(results=weekshedulerlegs)

#app.run(debug=True)
if __name__ == "__main__":
    app.run(debug=True)