import cv2

def compare(oldImage, newImage):
    # Load the images
    img1 = oldImage
    img2 = newImage

    # Check the dimensions
    if img1.shape != img2.shape:
        # Resize the images
        img1 = cv2.resize(img1, img2.shape[1::-1])

    # Convert the images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calculate the difference
    diff = cv2.absdiff(gray1, gray2)

    # Threshold the difference
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY)[1]

    # Calculate the percentage of pixels that differ
    percentage = (thresh.size - cv2.countNonZero(thresh)) / thresh.size * 100

    # Print the percentage
    print('Percentage of pixels that differ:', percentage)

    # Check if the images are the same
    if percentage < 3:
        print('The images are the same.')
    else:
        print('The images are different.')

    # Show the difference and thresholded images
    cv2.imshow('Difference', diff)
    cv2.imshow('Thresholded', thresh)
    if percentage < 3:
        return True
    else:
        return False
