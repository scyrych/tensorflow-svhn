import os

import PIL.Image
import PIL.ImageDraw
import pandas as pd

from unpacker import DigitStructWrapper


def display_bounding_boxes(img, bounding_boxes):
    """Displays an image and overlays the bounding boxes
        """
    # Opens and identifies the given image file
    img = PIL.Image.open(img)

    # Use draw module can be used to annotate the image
    draw = PIL.ImageDraw.Draw(img)
    b = bounding_boxes

    for i in range(len(b['label'])):
        # Bounding box rectangle [x0, y0, x1, y1]
        rectangle = ((b['left'][i], b['top'][i]),
                     (b['left'][i] + b['width'][i],
                      b['top'][i] + b['height'][i]))

        # Draw a rectangle on top of the image
        draw.rectangle(rectangle, outline="blue")

    # Return altered image
    img.show()
    return img


def display_bbox(image_path, bbox):
    img = PIL.Image.open(image_path)
    draw = PIL.ImageDraw.Draw(img)
    draw.rectangle(((bbox['x0'], bbox['y0']),
                    (bbox['x1'], bbox['y1'])),
                   outline='blue')
    img.show()
    return img


def process_bounding_boxes(bbox_file):
    df = _create_or_get_data_frame(bbox_file)
    df = _aggregate_data(df)
    df = _expand_bboxes(df)
    return df


def _create_or_get_data_frame(bbox_file):
    if not os.path.isfile(bbox_file):
        train_bbox = _get_bounding_boxes('data/train/digitStruct.mat')
        test_bbox = _get_bounding_boxes('data/test/digitStruct.mat')
        extra_bbox = _get_bounding_boxes('data/extra/digitStruct.mat')

        train_df = _dict_to_data_frame(train_bbox, 'data/train/')
        test_df = _dict_to_data_frame(test_bbox, 'data/test/')
        extra_df = _dict_to_data_frame(extra_bbox, 'data/extra/')

        print("Training", train_df.shape)
        print("Test", test_df.shape)
        print("Extra", extra_df.shape)
        print('')

        df = pd.concat([train_df, test_df, extra_df])

        print("Combined")

        df.to_csv(bbox_file, index=False)

        # Delete the old dataframes
        del train_df, test_df, extra_df, train_bbox, test_bbox, extra_bbox

    else:
        df = pd.read_csv(bbox_file)

    return df


def _get_bounding_boxes(start_path='.'):
    return DigitStructWrapper(start_path).unpack_all()


def _dict_to_data_frame(image_bounding_boxes, path):
    boxes = []

    for img in image_bounding_boxes:
        for bbox in img['boxes']:
            boxes.append({
                'filename': path + img['filename'],
                'label': bbox['label'],
                'width': bbox['width'],
                'height': bbox['height'],
                'top': bbox['top'],
                'left': bbox['left']})

    return pd.DataFrame(boxes)


def _aggregate_data(df):
    # Rename the columns to more suitable names
    df.rename(columns={'left': 'x0', 'top': 'y0'}, inplace=True)
    # Calculate x1 and y1
    df['x1'] = df['x0'] + df['width']
    df['y1'] = df['y0'] + df['height']
    # Perform the following aggregations
    aggregate = {'x0': 'min',
                 'y0': 'min',
                 'x1': 'max',
                 'y1': 'max',
                 'label': {
                     'labels': lambda x: list(x),
                     'num_digits': 'count'}}
    # Apply the aggration
    df = df.groupby('filename').agg(aggregate).reset_index()
    # Fix the column names after aggregation
    df.columns = [x[1] if x[0] == 'label' else x[0]
                  for i, x in enumerate(df.columns.values)]
    return df


def _expand_bboxes(df):
    df['x_increase'] = ((df['x1'] - df['x0']) * 0.3) / 2.0
    df['y_increase'] = ((df['y1'] - df['y0']) * 0.3) / 2.0

    df['x0'] = (df['x0'] - df['x_increase']).astype('int')
    df['y0'] = (df['y0'] - df['y_increase']).astype('int')
    df['x1'] = (df['x1'] + df['x_increase']).astype('int')
    df['y1'] = (df['y1'] + df['y_increase']).astype('int')

    return df
