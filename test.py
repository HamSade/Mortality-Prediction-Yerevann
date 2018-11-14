# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 11:23:15 2018

@author: hamed
"""


#%% Testing
              
#elif args.mode == 'test':
#
#    # ensure that the code uses test_reader
#    del train_reader
#    del val_reader
#    del train_raw
#    del val_raw
#
#    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
#                                            listfile=os.path.join(args.data, 'test_listfile.csv'),
#                                            period_length=48.0)
#    ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
#                          return_names=True)
#
#    data = ret["data"][0]
#    labels = ret["data"][1]
#    names = ret["names"]
#
#    predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
#    predictions = np.array(predictions)[:, 0]
#    metrics.print_metrics_binary(labels, predictions)
#
#    path = os.path.join(args.output_dir, "test_predictions", os.path.basename(args.load_state)) + ".csv"
#    utils.save_results(names, predictions, labels, path)
#
#else:
#    raise ValueError("Wrong value for args.mode")
  

#%% Evaluation
# ==========
# we simply feed the decoder's predictions back to itself for each step.
# Every time it predicts a word we add it to the output string, and if it
# predicts the EOS token we stop there. We also store the decoder's
# attention outputs for display later.

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


#%% ###################################################################
# We can evaluate random sentences from the training set and print out the
# input, target, and output to make some subjective quality judgements:

#def evaluateRandomly(encoder, decoder, n=10):
#    for i in range(n):
#        pair = random.choice(pairs)
#        print('>', pair[0])
#        print('=', pair[1])
#        output_words, attentions = evaluate(encoder, decoder, pair[0])
#        output_sentence = ' '.join(output_words)
#        print('<', output_sentence)
#        print('')


######################################################################
# Training and Evaluating
# =======================
#
# With all these helper functions in place (it looks like extra work, but
# it makes it easier to run multiple experiments) we can actually
# initialize a network and start training.
#
# Remember that the input sentences were heavily filtered. For this small
# dataset we can use relatively small networks of 256 hidden nodes and a
# single GRU layer. After about 40 minutes on a MacBook CPU we'll get some
# reasonable results.
#
# .. Note::
#    If you run this notebook you can train, interrupt the kernel,
#    evaluate, and continue training later. Comment out the lines where the
#    encoder and decoder are initialized and run ``trainIters`` again.
#

#hidden_size = 256
#encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
#attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
#
#trainIters(encoder1, attn_decoder1, 75000, print_every=5000)

######################################################################
#

#evaluateRandomly(encoder1, attn_decoder1)


######################################################################
# Visualizing Attention
# ---------------------
#
# A useful property of the attention mechanism is its highly interpretable
# outputs. Because it is used to weight specific encoder outputs of the
# input sequence, we can imagine looking where the network is focused most
# at each time step.
#
# You could simply run ``plt.matshow(attentions)`` to see attention output
# displayed as a matrix, with the columns being input steps and rows being
# output steps:
#

#output_words, attentions = evaluate(
#    encoder1, attn_decoder1, "je suis trop froid .")
#plt.matshow(attentions.numpy())


######################################################################
# For a better viewing experience we will do the extra work of adding axes
# and labels:
#

#def showAttention(input_sentence, output_words, attentions):
#    # Set up figure with colorbar
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    cax = ax.matshow(attentions.numpy(), cmap='bone')
#    fig.colorbar(cax)
#
#    # Set up axes
#    ax.set_xticklabels([''] + input_sentence.split(' ') +
#                       ['<EOS>'], rotation=90)
#    ax.set_yticklabels([''] + output_words)
#
#    # Show label at every tick
#    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
#
#    plt.show()
#
#
#def evaluateAndShowAttention(input_sentence):
#    output_words, attentions = evaluate(
#        encoder1, attn_decoder1, input_sentence)
#    print('input =', input_sentence)
#    print('output =', ' '.join(output_words))
#    showAttention(input_sentence, output_words, attentions)
#
#
#evaluateAndShowAttention("elle a cinq ans de moins que moi .")
#
#evaluateAndShowAttention("elle est trop petit .")
#
#evaluateAndShowAttention("je ne crains pas de mourir .")
#
#evaluateAndShowAttention("c est un jeune directeur plein de talent .")