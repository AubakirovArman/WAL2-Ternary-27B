"""E020 A/B uses exactly the explicit final E018 parent provenance resolver."""
import train_reasoning_qat as qat
from train_e019 import resolve_e019_parent

if __name__=='__main__':
    qat.resolve_parent=resolve_e019_parent
    qat.main()
