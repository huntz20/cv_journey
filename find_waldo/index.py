import cv2

def find_waldo() :
    waldo_ref = cv2.imread('./images/waldo.png', 0)
    where_waldo = cv2.imread('images/Where\'s Waldo_0.jpg')




    beach_gray = cv2.cvtColor(where_waldo, cv2.COLOR_BGR2GRAY)


    result = cv2.matchTemplate(beach_gray, waldo_ref, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    #Create a rectangle around the matched region
    waldo_height, waldo_width = waldo_ref.shape[:2]
    top_left = max_loc
    bottom_right = (top_left[0] + waldo_width, top_left[1] + waldo_height)
    cv2.rectangle(where_waldo, top_left, bottom_right, (0, 255, 0), 2)
    cv2.imshow('Waldo Found', where_waldo)

    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()