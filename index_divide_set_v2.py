import random
import os
import shutil
import cv2

original_paths = ['./crawled_mixed', './captures']
annotation_paths = ['./crawl_annotation', './capture_annotation']
dest_path = './datasets/oranges'
ratio = {'train': 0.8, 'test': 0.05, 'val': 0.15}

if __name__ == '__main__':
    # For label indicators
    test_text = ''
    train_text = ''
    val_text = ''

    # (Re)create the dataset directories
    try:
        shutil.rmtree(dest_path)
    except FileNotFoundError:
        pass
    os.makedirs(dest_path)
    os.makedirs(f"{dest_path}/images")
    os.makedirs(f"{dest_path}/images/test")
    os.makedirs(f"{dest_path}/images/train")
    os.makedirs(f"{dest_path}/images/val")
    os.makedirs(f"{dest_path}/labels")
    os.makedirs(f"{dest_path}/labels/test")
    os.makedirs(f"{dest_path}/labels/train")
    os.makedirs(f"{dest_path}/labels/val")

    idx = 1
    train_ratio = ratio['train'] / sum(ratio)
    test_ratio = ratio['test'] / sum(ratio) + train_ratio

    # zip for multiple input sets if needed
    for original_path, annotation_path in zip(original_paths, annotation_paths):
        for n in os.listdir(original_path):
            fname = n.split(".")[0]
            rand = random.random()  # floating num 0 → 1
            if rand < train_ratio:
                sub_path = 'train'
                train_text += f'./images/{sub_path}/{idx:06d}.jpg\n'
            elif rand < test_ratio:
                sub_path = 'test'
                test_text += f'./images/{sub_path}/{idx:06d}.jpg\n'
            else:
                sub_path = 'val'
                val_text += f'./images/{sub_path}/{idx:06d}.jpg\n'

            # Copy images to the new place
            cv2.imwrite(f'{dest_path}/images/{sub_path}/{idx:06d}.jpg', cv2.imread(f'{original_path}/{n}'))

            # Copy and rename the annotation if exists
            if os.path.isfile(f'{annotation_path}/{fname}.txt'):
                shutil.copy(f'{annotation_path}/{fname}.txt', f'./')
                os.rename(f'./{fname}.txt', f'./{idx:06d}.txt')
                shutil.move(f'./{idx:06d}.txt', f'{dest_path}/labels/{sub_path}')
            print(f"Screened #{idx:06d} {fname}", end="\r")
            idx += 1

    # Write indicators
    with open(f'{dest_path}/train.txt', 'w') as f:
        f.write(train_text[:-1])
    with open(f'{dest_path}/test.txt', 'w') as f:
        f.write(test_text[:-1])
    with open(f'{dest_path}/val.txt', 'w') as f:
        f.write(val_text[:-1])

    print()