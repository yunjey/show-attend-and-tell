from PIL import Image
import os


def main():
  splits = ['train', 'val']
  for split in splits:
    resized_folder = './image/%s2014_resized/' %split
    if not os.path.exists(resized_folder):
      os.makedirs(resized_folder)
    print 'Start resizing %s images.' %split
    folder = './image/%s2014' %split
    image_ids = os.listdir(folder)
    num_images = len(image_ids)
    for i, image_id in enumerate(image_ids):
      with open(os.path.join(folder, image_id), 'r+b') as f:
        with Image.open(f) as image:
          image=image.resize([224, 224], Image.ANTIALIAS)
          image.save(os.path.join(resized_folder, image_id), image.format)
          if i % 500 == 0:
            print 'Resized images: %d/%d' %(i, num_images)
            
if __name__ == '__main__':
  main()