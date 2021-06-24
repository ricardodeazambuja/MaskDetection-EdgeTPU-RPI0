import time
from io import BytesIO
import numpy as np

from picamera import PiCamera


from PIL import Image
from PIL import ImageDraw

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


DEBUG = True
THRESHOLD = 0.65 # current model has a tendency to hallucinate and see masks on white backgrounds :D

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
    common.set_input(mask_interpreter, image)
    mask_interpreter.invoke()
    mask_objs = detect.get_objects(mask_interpreter, threshold, (1.0,1.0))
    return mask_objs
    
print("Initializing EdgeTPU... mask model")
mask_labels = read_label_file("mask_labels.txt")
mask_interpreter = make_interpreter("ssdlite_mobiledet_mask_edgetpu.tflite")
mask_interpreter.allocate_tensors()
mask_input_details = mask_interpreter.get_input_details()
mask_output_details = mask_interpreter.get_output_details()

mask_objs = []
print("Starting...")
with open('detections.txt','w') as dfile:
        # Start the video process
        stream = BytesIO()    
        time.sleep(2)
        try:
            with PiCamera() as camera:
                camera.resolution = (640, 480)
                camera.framerate = 30
                camera.video_stabilization = True
                camera.video_denoise = True
                for i,_ in enumerate(camera.capture_continuous(stream,
                                                     format='rgb',
                                                     use_video_port=True,
                                                     resize=(320, 320))):

                    start = time.perf_counter()
                    stream.truncate()
                    stream.seek(0)
                    image = np.frombuffer(stream.getvalue(), dtype=np.uint8).reshape((320,320,3))
                    if image.size:
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
            if DEBUG and image.size:
                image = Image.fromarray(image)
                dimage = ImageDraw.Draw(image)
                if len(mask_objs):
                    draw_objects(dimage, mask_objs, mask_labels)
                image.save("/home/pi/final_img_mask.jpg")
                print("/home/pi/final_img_mask.jpg saved!")
            
            print("Done!")
