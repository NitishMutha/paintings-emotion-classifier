import tensorflow as tf, sys
from os import listdir
import csv

#files = listdir(sys.argv[1])
dir_path = 'data/test/'
files = listdir(dir_path)

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
               in tf.gfile.GFile("tf_files/retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

image_data = []
for imagefile in files:
    # change this as you see fit
    image_path = dir_path + imagefile

    # Read in the image_data
    image_data.append(tf.gfile.FastGFile(image_path, 'rb').read())

with open('result.csv','w', newline='') as csvfile:
    fieldnames = ['image', 'fear', 'happy','neutral','anger','sad']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()


    with tf.Session() as sess:
        count = 0
        for img in image_data:
            # Feed the image_data as input to the graph and get first prediction
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': img})

            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

            perdict_row = {}
            perdict_row['image'] = files[count]
            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                print('%s (score = %.5f)' % (human_string, score))
                if human_string == 'fear':
                    perdict_row['fear'] = score
                elif human_string == 'happy':
                    perdict_row['happy'] = score
                elif human_string == 'neutral':
                    perdict_row['neutral'] = score
                elif human_string == 'anger':
                    perdict_row['anger'] = score
                elif human_string == 'sad':
                    perdict_row['sad'] = score
            writer.writerow(perdict_row)
            count += 1
