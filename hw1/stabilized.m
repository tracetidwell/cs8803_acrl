function stable = stabilized(x, y)

stable = sum((x-y).^2) < 0.001;