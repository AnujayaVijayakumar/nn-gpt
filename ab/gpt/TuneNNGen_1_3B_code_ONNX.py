import ab.gpt.TuneNNGen as TuneNNGen


def main():
    TuneNNGen.main(llm_conf='ds_coder_1.3b_instruct.json', max_new_tokens=4 * 1024, onnx_run=True)


if __name__ == '__main__':
    main()
