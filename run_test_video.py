import tensorflow as tf
import os, random, utils, argparse, time
import transform
from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.video.io.ffmpeg_writer as ffmpeg_writer
import numpy as np
import config as cfg

BATCH_SIZE = 4
TMP_DIR = '.fns_frames_%s/' % random.randint(0,99999)

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of 'Perceptual Losses for Real-Time Style Transfer and Super-Resolution'"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--style_model', type=str, default=cfg.test_video['style_model'], help='location for model file (*.ckpt)')

    parser.add_argument('--content', type=str, default=cfg.test_video['content'],
                        help='File path of content image (notation in the paper : x)')

    parser.add_argument('--output', type=str, default=cfg.test_video['output'],
                        help='File path of output image (notation in the paper : y_c)')

    parser.add_argument('--max_size', type=int, default=None, help='The maximum width or height of input images')
    
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='The batch size')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    try:
        #Tensorflow r0.12 requires 3 files related to *.ckpt
        assert os.path.exists(args.style_model + '.index') and os.path.exists(args.style_model + '.meta') and os.path.exists(
            args.style_model + '.data-00000-of-00001')
    except:
        print('There is no %s'%args.style_model)
        print('Tensorflow r0.12 requires 3 files related to *.ckpt')
        print('If you want to restore any models generated from old tensorflow versions, this assert might be ignored')
        return None

    # --content
    try:
        assert os.path.exists(args.content)
    except:
        print('There is no %s' % args.content)
        return None

    # --max_size
    try:
        if args.max_size is not None:
            assert args.max_size > 0
    except:
        print('The maximum width or height of input image must be positive')
        return None

    # --output
    dirname = os.path.dirname(args.output)
    try:
        if len(dirname) > 0:
            os.stat(dirname)
    except:
        os.mkdir(dirname)

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # load content image
    vid = VideoFileClip(args.content, audio=False)
    video_writer = ffmpeg_writer.FFMPEG_VideoWriter(
        args.output, vid.size, vid.fps,
        audiofile=args.content, threads=None, ffmpeg_params=None)

    g = tf.Graph()
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device('/gpu:0'), \
            tf.Session(config=soft_config) as sess:
        batch_shape = (args.batch_size, vid.size[1], vid.size[0], 3)
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')

        preds = transform.Transform().net(img_placeholder)
        saver = tf.train.Saver()
        # if os.path.isdir(checkpoint_dir):
        #     ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        #     if ckpt and ckpt.model_checkpoint_path:
        #         saver.restore(sess, ckpt.model_checkpoint_path)
        #     else:
        #         raise Exception("No checkpoint found...")
        # else:
            # saver.restore(sess, checkpoint_dir)
        saver.restore(sess, args.style_model)

        X = np.zeros(batch_shape, dtype=np.float32)

        def style_and_write(count):
            for i in range(count, args.batch_size):
                X[i] = X[count - 1]  # Use last frame to fill X
            _preds = sess.run(preds, feed_dict={img_placeholder: X})
            for i in range(0, count):
                video_writer.write_frame(np.clip(_preds[i], 0, 255).astype(np.uint8))

        frame_count = 0  # The frame count that written to X
        curr = 0
        for frame in vid.iter_frames():
            X[frame_count] = frame
            frame_count += 1
            curr += 1
            if frame_count == args.batch_size:
                style_and_write(frame_count)
                frame_count = 0
                print( '{}%'.format( (curr / int(vid.fps * vid.duration)) * 100 ) )

        if frame_count != 0:
            style_and_write(frame_count)

        video_writer.close()

if __name__ == '__main__':
    main()
