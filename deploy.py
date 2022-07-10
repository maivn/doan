import streamlit as st
from streamlit.scriptrunner.script_run_context import get_script_run_ctx
from streamlit.scriptrunner import add_script_run_ctx
import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
import time
from PIL import Image


mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

model = tf.keras.models.load_model("model.h5")

# label = "Warmup...."

Action = ["STOP",
          "THIS MARSHALLER",
          "PROCEED TO NEXT MARSHALLER ON THE RIGHT",
          "PROCEED TO NEXT MARSHALLER ON THE LEFT",
          "PERSONNEL APPROACH AIRCRAFT ON THE RIGHT",
          "PERSONNEL APPROACH AIRCRAFT ON THE LEFT",
          "NORMAL",
          "TURN TO THE LEFT",
          "TURN TO THE RIGHT",
          "SLOW DOWN",
          "MOVE FORWARD"
          ]

st.title('Vo Nhu Mai graduation thesis')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Marshaller Signal Recognition')
st.sidebar.subheader('App modes')


@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        # print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return img


def detectpose(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    print(lm_list.shape)
    results = model.predict(lm_list)
    xacsuat = results.tolist()
    xacsuat = xacsuat[0]
    hanhdong = xacsuat.index(max(xacsuat))
    print(max(xacsuat))
    if max(xacsuat) < 0.9988:
        label = "Detecting..."
        return label
    label = Action[hanhdong]
    return label


app_mode = st.sidebar.selectbox('Please Select',
                                ['About My Project','Detect signal','Signal check']
                                )

if app_mode == 'About My Project':
    st.markdown(
        'In this application we are using **MediaPipe** for creating a Pose Track Points, LSTM to detect signal. \n'
        '**StreamLit** is to create the Web Graphical User Interface (GUI) ')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.video('https://www.youtube.com/watch?v=5KCU1mKV1Kk')

    st.markdown('''
          # About Me \n 
            I am ** Vo Nhu Mai ** from class **17ĐHKT01**. \n

            This is my graduation thesis \n

            This application will check the movement of aircraft ramp marshaller. \n
            There are 2 modes: Detect Signals and Check Signal. \n

            So choose the mode you want and have fun!!

            ''')
elif app_mode == 'Detect signal':

    st.sidebar.markdown('---')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown('---')

    st.markdown(' ## Output')

    stframe = st.empty()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        st.markdown('**Error: Cannot open camera**')

    prevTime = 0
    i = 0

    kpi1, kpi2 = st.columns(2)

    with kpi1:
        st.markdown("**FPS**")
        kpi1_text = st.markdown("0")
    label = ""
    with kpi2:
        st.markdown("**Signal**")
        kpi2_text = st.markdown(label)

    st.markdown("<hr/>", unsafe_allow_html=True)

    lm_list = []
    while True:

        # success, img = cap.read()
        i += 1
        ret, img = cap.read()
        if not ret:
            continue
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)

        n_time_steps = 40
        warmup_frames = 0
        print('aaaaaa')

        if i > warmup_frames:
            if results.pose_landmarks:
                c_lm = make_landmark_timestep(results)
                lm_list.append(c_lm)

                print(lm_list)
                print('bbbbbb', len(lm_list))
                if len(lm_list) == n_time_steps:
                    # Nhận diện
                    print('cccccc')
                    thread1 = threading.Thread(target=detectpose, args=(model, lm_list,))
                    add_script_run_ctx(thread1)
                    thread1.start()
                    lm_list = []

                img = draw_landmark_on_image(mpDraw, results, imgRGB)
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
        cv2.imshow("Image", imgRGB)

        # Dashboard
        kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
        kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{label}</h1>", unsafe_allow_html=True)

        imgRGB = image_resize(image=imgRGB, width=720, height=1280)
        stframe.image(imgRGB, channels='RGB', use_column_width=True)


    cap.release()

elif app_mode == 'Signal check':

    st.sidebar.markdown('---')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        '''
        Please enter one of the signal: \n
        STOP \n
        THIS MARSHALLER \n
        PROCEED TO NEXT MARSHALLER ON THE RIGHT \n
        PROCEED TO NEXT MARSHALLER ON THE LEFT \n
        PERSONNEL APPROACH AIRCRAFT ON THE RIGHT \n
        PERSONNEL APPROACH AIRCRAFT ON THE LEFT \n
        NORMAL \n
        TURN TO THE LEFT \n
        TURN TO THE RIGHT \n
        SLOW DOWN \n
        MOVE FORWARD \n
        ''')

    st.sidebar.markdown('---')

    st.markdown(' ## Output')

    stframe = st.empty()

    dongtac = st.text_input("Enter the signals", type="default")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        st.markdown('**Error: Cannot open camera**')

    prevTime = 0
    i = 0

    kpi1, kpi2 = st.columns(2)

    with kpi1:
        st.markdown("**FPS**")
        kpi1_text = st.markdown("0")
    label = ""

    with kpi2:
        st.markdown("**Signal**")
        kpi2_text = st.markdown(label)

    st.markdown("<hr/>", unsafe_allow_html=True)

    lm_list = []
    while True:

        # success, img = cap.read()
        i += 1
        ret, img = cap.read()
        if not ret:
            continue
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)

        n_time_steps = 40
        warmup_frames = 0
        print('aaaaaa')

        if i > warmup_frames:
            if results.pose_landmarks:
                c_lm = make_landmark_timestep(results)
                lm_list.append(c_lm)

                if len(lm_list) == n_time_steps:
                    # Nhận diện
                    thread1 = threading.Thread(target=detectpose, args=(model, lm_list,))
                    add_script_run_ctx(thread1)
                    thread1.start()
                    lm_list = []

                img = draw_landmark_on_image(mpDraw, results, imgRGB)
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
        cv2.imshow("Image", imgRGB)

        check = ""
        if dongtac == label:
            check = "GOOD JOB"
        else:
            check = "Keep trying"

        # Dashboard
        kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
        kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{check}</h1>", unsafe_allow_html=True)

        imgRGB = image_resize(image=imgRGB, width=720, height=1280)
        stframe.image(imgRGB, channels='RGB', use_column_width=True)


    cap.release()