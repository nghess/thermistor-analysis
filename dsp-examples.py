from scipy import zeros, signal, random
import matplotlib.pyplot as plt

# Trying out examples from the internet.


def filter_sbs():
    data = random.random(2000)
    b = signal.firwin(500, 0.004)
    z = signal.lfilter_zi(b, 1) * data[0]
    result = zeros(data.size)
    for i, x in enumerate(data):
        result[i], z = signal.lfilter(b, 1, [x], zi=z)
    return result


if __name__ == '__main__':
    result = filter_sbs()

data = filter_sbs()

x = [i for i in range(len(data))]
fig = plt.plot(x, data)

plt.show()