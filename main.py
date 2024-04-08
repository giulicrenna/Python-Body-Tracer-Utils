

def test_model() -> None:
    video = cv.VideoCapture(0)
    hands = HandDetector(static_image_mode=False ,max_hands=4, min_det_confidence=.1)

    while True:
        success, img = video.read()
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        positions: dict = hands.predict_hands_position(rgb)
        
        print(positions)
        
        cv.imshow("video", cv.cvtColor(rgb, cv.COLOR_RGB2BGR))
        if cv.waitKey(1) != -1:
            cv.destroyAllWindows()
            break

if __name__ == '__main__':
    test_model()
    #create_dataset_on_videocapture('HANDS_GUN')