from config import load_cmd_line
from utils import check_mind_set


if __name__ == "__main__":
    cmd_args = load_cmd_line()
    mind_type = cmd_args.get("mind_type")
    data_dir = cmd_args.get("data_dir")
    check_mind_set(data_dir=data_dir, mind_type=mind_type)
