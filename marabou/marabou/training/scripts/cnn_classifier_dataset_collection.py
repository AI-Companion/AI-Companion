import requests
from cv2 import cv2
import os
import glob
from marabou.commons import DATA_DIR

def main():
    classes = ["sunglasses", "jeans", "dress"]
    dataset_dir = os.path.join(DATA_DIR, "clothing_classifier")
    if not os.path.exists(dataset_dir):
        raise ValueError("please create dataset folder and place download scripts inside it")

    for cl in classes:
        class_dir = os.path.join(dataset_dir, cl)
        status_file = os.path.join(class_dir, 'status.txt')
        total = 0
        collect_images = False
        if not os.path.isfile(os.path.join(dataset_dir, cl + ".txt")):
            raise ValueError("no %s file" % cl)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
            with open(status_file, 'w') as f:
                f.write('NOK')
        if not os.path.isfile(status_file):
            collect_images = True
            with open(status_file, 'w') as f:
                f.write('NOK')
        class_status = open(status_file).read().strip()
        if class_status != "OK":
            collect_images = True
        download_file = os.path.join(dataset_dir, cl + ".txt")
        if not collect_images:
            print("[INFO] {} images already collected".format(cl))
        else:
            print("[INFO] collecting {} images...".format(cl))
            files = glob.glob(os.path.join(class_dir,"*"))
            for f in files:
                os.remove(f)
            rows = open(download_file).read().strip().split("\n")
            for url in rows:
                try:
                    # try to download the image
                    r = requests.get(url, timeout=60)
                    # save the image to disk
                    p = os.path.sep.join([class_dir, "{}.jpg".format(
                        str(total).zfill(8))])
                    f = open(p, "wb")
                    f.write(r.content)
                    f.close()
                    # update the counter
                    total += 1
                # handle if any exceptions are thrown during the download process
                except:
                    print("[INFO] error downloading {}...skipping".format(p))
            print("[INFO] collected {} images".format(total))
            with open(status_file, 'w') as f:
                f.write('OK')
        # loop over the image paths we just downloaded
        image_list = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        image_list.remove("status.txt")
        total = 0
        print("[INFO] checking corrupted images")
        for im in image_list:
            image_path = os.path.join(class_dir, im)
            # initialize if the image should be deleted or not
            delete = False
            # try to load the image
            try:
                image = cv2.imread(image_path)
                # if the image is `None` then we could not properly load it
                # from disk, so delete it
                if image is None:
                    delete = True
                    total += 1
            # if OpenCV cannot load the image then the image is likely
            # corrupt so we should delete it
            except:
                print("Except")
                delete = True
            # check to see if the image should be deleted
            if delete:
                print("[INFO] deleting {}".format(image_path))
                os.remove(image_path)
        print("[INFO] deleted {} images".format(total))

if __name__ == '__main__':
    main()
