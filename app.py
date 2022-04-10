from poses.main import *
from flask import Flask, render_template, Response

app = Flask(__name__)
app.config['ENV'] = 'development'
app.config['DEBUG'] = True
app.config['TESTING'] = True


cap = cv2.VideoCapture(0)


def start():
    images = load_images_from_folder(folder)
    total_cnt = len(images)
    cnt = 0

    cv2.imshow('PregFit Tutorial', images[cnt])
    while cap.isOpened():
        # Capture the video frame
        ret, frame = cap.read()
        testImage = asarray(images[cnt])
        threshold = 0.09
        if check_matching(frame, testImage, threshold):
            print("matched.........")
            playsound('poses\\beep.wav')
            cnt += 1

            if cnt == total_cnt:
                print("completed ")
                break
            get_skeleton(asarray(images[cnt]))
            cv2.imshow('tutorial', images[cnt])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()


@app.route('/poses', methods=['GET', 'POST'])
def poses():

    return Response(start(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
print("Yay")
