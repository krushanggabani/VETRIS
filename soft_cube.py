import argparse
import vetris as vt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    args = parser.parse_args()

    print(args)


    cfg = vt.init(theme="dark", backend="cpu")
    print("Initialized:", cfg)




if __name__ == "__main__":
    main()

