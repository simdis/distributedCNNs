import papermill as pm
import argparse


def define_and_parse_flags(parse=True):
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=int, default=1,
                        help='The number of CNNs.')
    parser.add_argument('--N', type=int, default=50,
                        help='The number of nodes.')
    parser.add_argument('--xm', type=float, default=15,
                        help='Space width as (-xm, xm).')
    parser.add_argument('--dt', type=float, default=7.5,
                        help='Transmission range.')
    parser.add_argument('--datarate', type=float, default=9241.6,
                        help='Datarate (measured in KB/s --> default is 72.2 Mb/s == 9241.6 KB/s)')
    parser.add_argument('--Ks', type=int, default=227 * 227 * 3 * 4 / 1024,
                        help='Image Size (measured in KB --> default is a floating-point RGB image of size 227x227).')
    parser.add_argument('--orange_p', type=float, default=0.45,
                        help='OrangePi Zero percentage.')
    parser.add_argument('--beagle_p', type=float, default=0.45,
                        help='BeagleBone AI percentage.')
    parser.add_argument('--pi3_p', type=float, default=0.10,
                        help='Raspberry PI 3B+ percentage.')
    parser.add_argument('--cnn_name', type=str, default='alex',
                        help='The name of the CNN to place.')

    parser.add_argument('--output_dir', required=True,
                        help='The folder in which the execute_notebook are stored. A folder named results with all the csvs will be created.')
    parser.add_argument('--time_limit', type=float, default=300.0,
                        help='GUROBI solver time limit.')

    parser.add_argument('--num_exps', type=int, default=500,
                        help='The number of exps to run.')

    # Return the parser or the parsed values according to the parameter 'parse'.
    if parse:
        return parser.parse_args()
    return parser


if __name__ == '__main__':
    FLAGS = define_and_parse_flags()

    for i in range(FLAGS.num_exps):
        print('EXP {}'.format(i))
        pm.execute_notebook(
            'Multi Source Multi CNN-Version To Be Executed.ipynb',
            '{}/exp{}.ipynb'.format(FLAGS.output_dir, i),
            parameters=dict(
                exp_id=i,
                C=FLAGS.C,
                N=FLAGS.N,
                xm=FLAGS.xm,
                dt=FLAGS.dt,
                datarate=FLAGS.datarate,
                Ks=FLAGS.Ks,
                orange_p=FLAGS.orange_p,
                beagle_p=FLAGS.beagle_p,
                pi3_p=FLAGS.pi3_p,
                cnn_name=FLAGS.cnn_name,
                output_dir='{}/results'.format(FLAGS.output_dir),
                time_limit=FLAGS.time_limit
            )
        )
