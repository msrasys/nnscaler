from cube.schedule.iterator import get_pipeline_seq_space_size

import argparse


def get_seq_space_size(nstage, nmb):
    """
    Calculate legal sequence number given num stage and num microbatch

    \prod \limits_{i=1}^{nmb} C(nstage, i*nstage)

    Args:
        nstage: number of stages
        nmb: number of micro batch
    
    Return:
        total legal line
    """
    return get_pipeline_seq_space_size(nstage, nmb)


def get_device_space_size(nstage, nmb, ndevice):
    """
    Calculate legal spatial sequence number given num stage and num microbatch

    \prod \limits_{i=1}^{nmb} C(nstage, i*nstage)

    Args:
        nstage: number of stages
        nmb: number of micro batch
        ndevice: number of device
    
    Return:
        total legal line
    """
    num_actions = nmb * nstage * 2
    device_space_size = ndevice ** num_actions
    return device_space_size


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--nstage', type=int, default=4,
                        help='number of stages')
    parser.add_argument('--nmb', type=int, default=4,
                        help='number of micro-batch')
    parser.add_argument('--ndev', type=int, default=4,
                        help='number of devices')
    args = parser.parse_args()

    seq_space = get_seq_space_size(args.nstage, args.nmb)
    print('legal sequence space: {}'.format(seq_space))
    dev_space = get_device_space_size(args.nstage, args.nmb, args.ndev)
    print('spatial space for one sequence: {}'.format(dev_space))
    total_space = seq_space * dev_space
    print('total space: {}'.format(total_space))
