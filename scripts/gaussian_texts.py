import numpy as np
from dask.distributed import Client
import json

if __name__ == "__main__":
    print("Starting client")

    per_label_mu_sigma = json.load(open("per_label_mu_sigma.json", "r"))

    # client = Client(n_workers=20) # In this example I have 8 cores and processes (can also use threads if desired)

    def my_function(vars, i):
        mu, sigma = vars
        out = []
        for _ in range(200):
            new_data_point = np.random.multivariate_normal(mu, sigma, 1)
            out.append([new_data_point, i])
        print("Done with {}".format(i))
        return out

    results = []

    for i in per_label_mu_sigma:
        print("Spawned {}".format(i))
        # future = client.submit(my_function, per_label_mu_sigma[i], i)
        future =my_function(per_label_mu_sigma[i], i)
        results.append(future)

    # results = client.gather(results)
    np.savez("gaussian_texts", results)
    # client.close()