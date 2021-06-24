import io
import time
import numpy as np
import picamera


from PIL import Image
from PIL import ImageDraw

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


DEBUG = True
THRESHOLD = 0.5

def draw_objects(draw, objs, labels, color='red'):
  """Draws the bounding box and label for each object."""
  for obj in objs:
    bbox = obj.bbox
    draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                   outline=color)
    draw.text((bbox.xmin + 10, bbox.ymin + 10),
              '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
              fill=color)
    
def process_img(image, threshold=0.5):
    _, mask_scale = common.set_resized_input(mask_interpreter, image.size, lambda size: image.resize(size, Image.NEAREST))
    mask_interpreter.invoke()
    mask_objs = detect.get_objects(mask_interpreter, threshold, mask_scale)
    return mask_objs
    
print("Initializing EdgeTPU... mask model")
mask_labels = read_label_file("mask_labels.txt")
mask_interpreter = make_interpreter("ssdlite_mobiledet_mask_edgetpu.tflite")
mask_interpreter.allocate_tensors()
mask_input_details = mask_interpreter.get_input_details()
mask_output_details = mask_interpreter.get_output_details()


print("Initializing picamera...")
# See https://picamera.readthedocs.io/en/release-1.13/api_camera.html
# for details about the parameters:
frameWidth = 256
frameHeight = 256
frameRate = 5
contrast = 40

# Set the picamera parametertaob
camera = picamera.PiCamera()
camera.resolution = (frameWidth, frameHeight)
camera.framerate = frameRate
# camera.contrast = contrast
camera.video_stabilization = True
camera.video_denoise = True
#camera.rotation = 180

print("Starting...")
with open('detections.txt','w') as dfile:
        # Start the video process
        stream = io.BytesIO()    
        time.sleep(2)
        try:
            for i,_ in enumerate(camera.capture_continuous(stream, format='rgb')):
                start = time.perf_counter()
                stream.truncate()
                stream.seek(0)
                image = Image.fromarray(np.frombuffer(stream.getvalue(), dtype=np.uint8).reshape((frameWidth,frameHeight,3)))
                if image:
                    curr_time = time.ctime()
                    mask_objs = process_img(image, threshold=THRESHOLD)
                    for obj in mask_objs:
                        output = f"{i};{start};{curr_time};{mask_labels.get(obj.id)};{obj.id};{obj.score};{obj.bbox}\n"
                        dfile.write(output)
                        if DEBUG:
                            print(output)
                print(f'{i}: {(time.perf_counter() - start) * 1000:.2f} ms\n')
        except KeyboardInterrupt:
            pass
        finally:
            if DEBUG:
                dimage = ImageDraw.Draw(image)
                if len(mask_objs):
                    draw_objects(dimage, mask_objs, mask_labels)
                image.save("/home/pi/final_img_mask.jpg")
                print("/home/pi/final_img_mask.jpg saved!")
            
            print("Done!")
