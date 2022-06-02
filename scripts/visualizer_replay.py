import os

import visdom
import argparse

DEFAULT_PORT = 8097
DEFAULT_HOSTNAME = "http://localhost"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replay Visualization Logs')
    parser.add_argument('-dir', metavar='dir', type=str, help='Directory of logs', required=True)
    parser.add_argument('-port', metavar='port', type=int, default=DEFAULT_PORT,
                        help='Port the visdom server is running on.')
    parser.add_argument('-server', metavar='server', type=str,
                        default=DEFAULT_HOSTNAME,
                        help='Server address of the target to run the demo on.')
    parser.add_argument('-base_url', metavar='base_url', type=str,
                        default='/',
                        help='Base Url.')
    FLAGS = parser.parse_args()
    viz = visdom.Visdom(port=FLAGS.port, server=FLAGS.server, base_url=FLAGS.base_url)

    if not os.path.exists(FLAGS.dir) and os.path.isdir(FLAGS.dir):
        raise Exception(f"Error: no directory at {FLAGS.dir}")

    if not viz.check_connection():
        raise Exception("Visdom server has to be started ahead of time")

    for filename in os.listdir(FLAGS.dir):
        if filename.endswith(".vis") and os.path.isfile(os.path.join(FLAGS.dir, filename)):
            print(f"Replaying {os.path.join(FLAGS.dir, filename)}")
            viz.replay_log(os.path.join(FLAGS.dir, filename))

    print("All Logs Replayed")
