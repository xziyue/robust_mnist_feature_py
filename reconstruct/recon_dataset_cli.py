import click
import pickle
import sys

#from reconstruct_nonrobust_feature import reconstruct_feature as recon_nonrobust

def _recon_robust_dataset_task(left, right, outname):
    import numpy as np
    np.random.seed(0x197de381)
    from reconstruct.reconstruct_robust_feature import reconstruct_feature as recon_robust

    result = []
    numDone = 0
    numTotal = right - left

    for i in range(left, right):
        numDone = (i - left)
        res = recon_robust(i)
        result.append(res)
        print('{}/{}'.format(numDone, numTotal))

    with open(outname, 'wb') as outfile:
        pickle.dump(result, outfile)


def _recon_nonrobust_dataset_task(left, right, outname):
    import numpy as np
    np.random.seed(0x197de381)
    from reconstruct.reconstruct_nonrobust_feature import reconstruct_feature as recon_nonrobust

    result = []
    numDone = 0
    numTotal = right - left

    for i in range(left, right):
        numDone = (i - left)
        res = recon_nonrobust(i)
        result.append(res)
        print('{}/{}'.format(numDone, numTotal))
        sys.stdout.flush()


    with open(outname, 'wb') as outfile:
        pickle.dump(result, outfile)

@click.group()
def main():
    pass

@main.command()
@click.argument('left')
@click.argument('right')
@click.argument('outname')
def robust(left, right, outname):
    left = int(left)
    right = int(right)
    assert right > left
    _recon_robust_dataset_task(left, right, outname)

@main.command()
@click.argument('left')
@click.argument('right')
@click.argument('outname')
def nonrobust(left, right, outname):
    left = int(left)
    right = int(right)
    assert right > left
    _recon_nonrobust_dataset_task(left, right, outname)



if __name__ == '__main__':
    main()