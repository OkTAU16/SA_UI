

if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    img = cv2.imread("SLM_logo.png")
    img = cv2.resize(img, (1024, 200))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct color display
    cv2.imwrite("SLM_logo_short.png", img)
