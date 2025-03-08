import csv

class History:
    def __init__(self):
        self.history = {
            'iteration': [],
            'm_ap': [],
            'Step': [],
            'U_MIL_loss': [],
            'MA_loss': [],
            'M_MIL_loss': [],
            'Triplet_loss': [],
            'LR': []
        }

    def update(self, iteration, m_ap, Step, U_MIL_loss, MA_loss, M_MIL_loss, Triplet_loss, LR):
        self.history['iteration'].append(iteration)
        self.history['m_ap'].append(m_ap)
        self.history['Step'].append(Step)
        self.history['U_MIL_loss'].append(U_MIL_loss.item())
        self.history['MA_loss'].append(MA_loss.item())
        self.history['M_MIL_loss'].append(M_MIL_loss.item())
        self.history['Triplet_loss'].append(Triplet_loss.item())
        self.history['LR'].append(LR)

    def save_to_csv(self, filepath):
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.history.keys())
            writer.writeheader()
            writer.writerows([dict(zip(self.history, t)) for t in zip(*self.history.values())])
