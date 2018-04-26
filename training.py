import sys
import time
from itertools import chain

import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from data import *
from etc import *
from network import *
import paths

parser = argparse.ArgumentParser(description='Unsupervised Voice Conversion')

# dataset params
parser.add_argument('--dataset_1', required=False, choices=['blizzard', 'ljspeech'], default='ljspeech')
parser.add_argument('--dataset_2', required=False, choices=['blizzard', 'ljspeech'], default='blizzard')
parser.add_argument('--sr', type=int, help='audio sampling rate', default=20000)
parser.add_argument('--num_freq', type=int, help='number of frequency used for stft', default=1024)
parser.add_argument('--num_freq_scaled', type=int, help='number of frequency scaled', default=256)


# learning algorithm params
parser.add_argument('--lr_gen', type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--lr_disc', type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--decay_rate', type=float, help='learning rate decaying rate', default=0.95)
parser.add_argument('--decay_step', type=int, help='learning rate decaying step', default=1000)
parser.add_argument('--epochs', type=int, help='training epochs', default=1)
parser.add_argument('--batch_size', type=int, default=1)

# training scheme
parser.add_argument('--disc_step', type=int, default=5)
parser.add_argument('--gan_loss_ratio', type=float, default=1.)

# etc
parser.add_argument('--cuda', type=str2bool, help='whether to use cuda(GPU)', default=True)
parser.add_argument('--save_step', type=int, help='steps btw model and image', default=1000)
parser.add_argument('--save_result', type=str2bool, default=True)

parser.add_argument('--model_name', type=str, default='unsupervised_vc')
parser.add_argument('--restore_step', type=int, help='Global step to restore checkpoint', default=0)


def discriminator_loss(d_real, d_fake):
    d_loss = 0.5 * (torch.mean((d_real - 1) ** 2) + torch.mean(d_fake ** 2))
    g_loss = 0.5 * torch.mean((d_fake - 1) ** 2)
    return d_loss, g_loss


def spec_loss(s_synth, s_target, n_priority_freq, cuda=True):
    loss = torch.abs(s_synth - s_target.transpose(1, 2))
    loss = 0.5 * torch.mean(loss) + 0.5 * torch.mean(loss[:, :n_priority_freq, :])
    if cuda:
        loss = loss.cuda()
    return loss


if __name__ == '__main__':
    args = parser.parse_args()

    dataset_tr_path = paths.DATA_BLIZZARD_TRAINING_PATH if args.dataset == 'blizzard' else paths.DATA_LJ_TRAINING_PATH
    dataset_va_path = paths.DATA_BLIZZARD_VALIDATION_PATH if args.dataset == 'blizzard' else paths.DATA_LJ_VALIDATION_PATH
    dataset_te_path = paths.DATA_BLIZZARD_TEST_PATH if args.dataset == 'blizzard' else paths.DATA_LJ_TEST_PATH

    tts_model_base_path = os.path.join(paths.MODEL_BASE_PATH, args.model_name_tts)
    asr_model_base_path = os.path.join(paths.MODEL_BASE_PATH, args.model_name_asr)
    model_base_path = os.path.join(paths.MODEL_BASE_PATH, args.model_name_unsup)
    if not os.path.exists(model_base_path):
        os.mkdir(model_base_path)

    result_base_path = os.path.join(paths.RESULT_PATH, args.model_name_unsup)
    tr_ev_result_path = os.path.join(result_base_path, paths.TRAINING_STR)
    va_ev_result_path = os.path.join(result_base_path, paths.VALIDATION_STR)
    te_ev_result_path = os.path.join(result_base_path, paths.TEST_STR)
    for result_path in [result_base_path, tr_ev_result_path, va_ev_result_path, te_ev_result_path]:
        if not os.path.exists(result_path):
            os.mkdir(result_path)

    # Get dataset
    text_tr = TextDataset(dataset_tr_path, args)
    wave_tr = AudioDataset(dataset_tr_path, args)
    parallel_tr = ParallelDataset(dataset_va_path, args, is_training=True)
    parallel_va = ParallelDataset(dataset_va_path, args, is_training=False)
    parallel_te = ParallelDataset(dataset_te_path, args, is_training=False)

    # Construct model
    modules_dict = {TEXT_ENC: TextEncoder(args),
                    WAVE_DEC: WaveDecoder(args),
                    ASR_MODULE: DeepSpeech(args),
                    Z_DISC: Discriminator(args, is_z_disc=True),
                    W_DISC: Discriminator(args, is_z_disc=False)}

    if args.cuda:
        for key, module in modules_dict.items():
            modules_dict[key] = torch.nn.DataParallel(module).cuda()

    if args.decoder == "beam":
        from network import BeamCTCDecoder

        with open(paths.DICTIONARY_PATH, 'r') as dict_open:
            labels = dict_open.readline().rstrip()

        decoder = BeamCTCDecoder(labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width, num_processes=args.lm_workers)
    else:
        decoder = None

    # Make optimizer
    optimizer_gen_1st, optimizer_disc_1st, optimizer_gen_2nd, optimizer_disc_2nd = _get_optimizers()

    # Load checkpoint if exists
    try:
        print('loading checkpoint from ' + model_base_path)
        checkpoint = torch.load(os.path.join(model_base_path, 'checkpoint_%d.pth.tar' % args.restore_step_unsup))
        for key in [TEXT_ENC, WAVE_DEC, ASR_MODULE, Z_DISC, W_DISC]:
            modules_dict[key].load_state_dict(checkpoint['%s_state_dict' % key])

        optimizer_gen_1st, optimizer_disc_1st, optimizer_gen_2nd, optimizer_disc_2nd = _get_optimizers()
        optimizer_gen_1st.load_state_dict(checkpoint['optimizer_gen_1st'])
        optimizer_disc_1st.load_state_dict(checkpoint['optimizer_disc_1st'])
        optimizer_gen_2nd.load_state_dict(checkpoint['optimizer_gen_2nd'])
        optimizer_disc_2nd.load_state_dict(checkpoint['optimizer_disc_2nd'])
        for opt in [optimizer_gen_1st, optimizer_disc_1st, optimizer_gen_2nd, optimizer_disc_2nd]:
            opt.param_groups[-1]['initial_lr'] = opt.param_groups[-1]['lr']
        print("\n--------model restored at step %d--------\n" % args.restore_step_tts)

    except:
        try:
            print('loading checkpoint from\n\t%s\n\t%s' % (tts_model_base_path, asr_model_base_path))
            checkpoint = torch.load(os.path.join(tts_model_base_path, 'checkpoint_%d.pth.tar' % args.restore_step_tts))
            for key in [TEXT_ENC, WAVE_DEC]:
                modules_dict[key].load_state_dict(checkpoint['%s_state_dict' % key])

            checkpoint = torch.load(os.path.join(asr_model_base_path, 'checkpoint_%d.pth.tar' % args.restore_step_asr))
            modules_dict[ASR_MODULE].load_state_dict(checkpoint['%s_state_dict' % ASR_MODULE])

            optimizer_gen_1st, optimizer_disc_1st, optimizer_gen_2nd, optimizer_disc_2nd = _get_optimizers()
            for opt in [optimizer_gen_1st, optimizer_disc_1st, optimizer_gen_2nd, optimizer_disc_2nd]:
                opt.param_groups[-1]['initial_lr'] = opt.param_groups[-1]['lr']
            print("\n--------model restored at step %d (TTS) and %d (ASR)--------\n"
                  % (args.restore_step_tts, args.restore_step_asr))

        except:
            print("\n--------Start New Training--------\n")

    optimizers_dict = {STEP_GEN_1ST: optimizer_gen_1st,
                       STEP_DISC_1ST: optimizer_disc_1st,
                       STEP_GEN_2ND: optimizer_gen_2nd,
                       STEP_DISC_2ND: optimizer_disc_2nd}

    print_modules_size(modules_dict)

    # define parameters to be updated, optimizers to be used and schedulers for learning rate control
    lr_scheds = {STEP_GEN_1ST: lr_scheduler.StepLR(optimizer_gen_1st, step_size=args.decay_step, gamma=args.decay_rate),
                 STEP_DISC_1ST: lr_scheduler.StepLR(optimizer_disc_1st, step_size=args.decay_step,
                                                    gamma=args.decay_rate),
                 STEP_GEN_2ND: lr_scheduler.StepLR(optimizer_gen_2nd, step_size=args.decay_step, gamma=args.decay_rate),
                 STEP_DISC_2ND: lr_scheduler.StepLR(optimizer_disc_2nd, step_size=args.decay_step,
                                                    gamma=args.decay_rate)}

    # Decide loss function
    criterion_l1 = nn.L1Loss()
    criterion_ctc = CTCLoss()
    if args.cuda:
        criterion_l1 = criterion_l1.cuda()
        criterion_ctc = criterion_ctc.cuda()

    best_losses = {key: np.inf for key in [TTS_LOSS, ASR_LOSS]}

    # Loss for frequency of human register
    n_priority_freq = int(3000 / (args.sr * 0.5) * args.num_freq)

    for epoch in range(args.epochs):
        # create shuffled dataloaders
        wave_loader = DataLoader(wave_tr, batch_size=args.batch_size, shuffle=True,
                                 drop_last=True, collate_fn=collate_fn_audio, num_workers=8)
        text_loader = DataLoader(text_tr, batch_size=args.batch_size, shuffle=True,
                                 drop_last=True, collate_fn=collate_fn_text, num_workers=8)

        log_counts = {key: 0 for key in [STEP_GEN_1ST, STEP_DISC_1ST, STEP_GEN_2ND, STEP_DISC_2ND]}

        dataloader_length = min(len(wave_loader), len(text_loader))
        for step in range(dataloader_length):
            current_step = step + args.restore_step_unsup + epoch * dataloader_length + 1

            # EVALUATION, parameters and result saving
            if current_step % args.save_step == 0 or current_step == 1:
                for key in modules_dict.keys():
                    modules_dict[key] = modules_dict[key].eval()

                for phase_str, parallel_dataset, result_path in zip(["VALIDATION", "TRAINING", "TEST"],
                                                                    [parallel_va, parallel_tr, parallel_te],
                                                                    [va_ev_result_path,
                                                                     tr_ev_result_path,
                                                                     te_ev_result_path]):

                    eval_data_loader = DataLoader(parallel_dataset, batch_size=args.batch_size, shuffle=False,
                                                  drop_last=False, collate_fn=collate_fn_parallel, num_workers=8)

                    curr_loss_dict = {key: 0. for key in [TTS_LOSS, ASR_LOSS, T_RECON_LOSS, W_RECON_LOSS,
                                                          Z_GAN_LOSS, Z_DISC_LOSS, W_GAN_LOSS, W_DISC_LOSS]}

                    batch_cnt = 0
                    for j, data in enumerate(eval_data_loader):
                        if phase_str == "TRAINING" and j >= 50:
                            break

                        batch_cnt = j

                        # get next batch from dataloader
                        mel, spec, T, CTC_target, CTC_len = [d for d in data]
                        T_for_TTS = _prepare_text_tensor(T, for_tts=True)
                        T_for_ASR = _prepare_text_tensor(T, for_tts=False)

                        mel, spec, T_for_TTS, T_for_ASR, CTC_target, CTC_len = \
                            [_as_variable(d) for d in [mel, spec, T_for_TTS, T_for_ASR, CTC_target, CTC_len]]

                        # Forward
                        z_text = modules_dict[TEXT_ENC](T_for_TTS)
                        mel_output, spec_output = modules_dict[WAVE_DEC](z_text, mel_input=None)
                        T_recon = modules_dict[ASR_MODULE](spec_output.transpose(1, 2))
                        T_output = modules_dict[ASR_MODULE](spec)

                        T_decoded = _decode_as_integers(T_output)

                        T_for_TTS2 = _as_variable(_prepare_text_tensor(T_decoded, for_tts=True))

                        z_text2 = modules_dict[TEXT_ENC](T_for_TTS2)

                        mel_recon, spec_recon = modules_dict[WAVE_DEC](z_text2, mel_input=None)

                        w_disc_output_fake = modules_dict[W_DISC](spec_output.transpose(1, 2))
                        w_disc_output_real = modules_dict[W_DISC](spec)
                        z_disc_output_fake = modules_dict[Z_DISC](z_text2)
                        z_disc_output_real = modules_dict[Z_DISC](z_text)

                        # Calculate loss
                        ctc_probsize = Variable(torch.IntTensor([T_output.size()[1]] * T_output.size()[0]))
                        ctc_probsize_recon = Variable(torch.IntTensor([T_recon.size()[1]] * T_recon.size()[0]))
                        if args.cuda:
                            ctc_probsize = ctc_probsize.cuda()
                            ctc_probsize_recon = ctc_probsize_recon.cuda()

                        min_length = min(mel.size()[1], args.audio_len)

                        temp_asr_loss = \
                            ctc_loss(T_output, CTC_target, ctc_probsize, CTC_len, criterion_ctc, args.batch_size, args.cuda) \
                            * args.scaling_text_decoding_loss
                        temp_tts_loss = \
                            mel_loss(mel_output[:, :, :min_length], mel[:, :min_length, :], criterion_l1, args.cuda) + \
                            spec_loss(spec_output[:, :, :min_length], spec[:, :min_length, :], n_priority_freq, args.cuda)
                        temp_t_recon_loss = \
                            ctc_loss(T_recon, CTC_target, ctc_probsize_recon, CTC_len, criterion_ctc, args.batch_size, args.cuda) \
                            * args.scaling_text_decoding_loss
                        temp_w_recon_loss = \
                            mel_loss(mel_recon[:, :, :min_length], mel[:, :min_length, :], criterion_l1, args.cuda) + \
                            spec_loss(spec_recon[:, :, :min_length], spec[:, :min_length, :], n_priority_freq, args.cuda)
                        temp_z_disc_loss, temp_z_gan_loss = discriminator_loss(z_disc_output_real, z_disc_output_fake)
                        temp_z_gan_loss *= args.scaling_gan_loss
                        temp_w_disc_loss, temp_w_gan_loss = discriminator_loss(w_disc_output_real, w_disc_output_fake)
                        temp_w_gan_loss *= args.scaling_gan_loss

                        curr_loss_dict[ASR_LOSS] += temp_asr_loss.data[0]
                        curr_loss_dict[TTS_LOSS] += temp_tts_loss.data[0]
                        curr_loss_dict[T_RECON_LOSS] += temp_t_recon_loss.data[0]
                        curr_loss_dict[W_RECON_LOSS] += temp_w_recon_loss.data[0]
                        curr_loss_dict[Z_GAN_LOSS] += temp_z_gan_loss.data[0]
                        curr_loss_dict[Z_DISC_LOSS] += temp_z_disc_loss.data[0]
                        curr_loss_dict[W_GAN_LOSS] += temp_w_gan_loss.data[0]
                        curr_loss_dict[W_DISC_LOSS] += temp_w_disc_loss.data[0]

                        # save result files
                        if j == 0:
                            result_epoch_path = os.path.join(result_path, str(current_step))
                            if not os.path.exists(result_epoch_path):
                                os.mkdir(result_epoch_path)

                            gt_text = _decode_as_text(T_for_ASR.data)
                            audio_writer(result_epoch_path, spec_output, name_head='TTS', use_cuda=args.cuda)
                            audio_writer(result_epoch_path, spec_recon, name_head='S_recon', use_cuda=args.cuda)
                            text_writer(result_epoch_path, _decode_as_text(T_decoded), gt_text,
                                        name_head='ASR', use_cuda=args.cuda)
                            text_writer(result_epoch_path, _decode_as_text(_decode_as_integers(T_recon)), gt_text,
                                        name_head='T_recon', use_cuda=args.cuda)

                    step_logs = '[EPOCH LOG - %s]\n' % phase_str
                    step_logs += 'Step: %d\n' % current_step

                    for loss_key in curr_loss_dict.keys():
                        step_logs += '%s loss: %.5f\n' % (loss_key, curr_loss_dict[loss_key] / (batch_cnt + 1))

                    step_logs += '--------------------------------------'
                    print(step_logs)
                    sys.stdout.flush()

                    if phase_str == "VALIDATION" and current_step > 1:
                        save = True

                        for loss_key in [TTS_LOSS, ASR_LOSS]:
                            if best_losses[loss_key] < curr_loss_dict[loss_key]:
                                save = False

                        if save:
                            # save model parameters
                            dict_to_save = {}
                            for key in [TEXT_ENC, WAVE_DEC, ASR_MODULE, Z_DISC, W_DISC]:
                                dict_to_save['%s_state_dict' % key] = modules_dict[key].state_dict()

                            dict_to_save['optimizer_gen_1st'] = optimizers_dict[STEP_GEN_1ST].state_dict()
                            dict_to_save['optimizer_disc_1st'] = optimizers_dict[STEP_DISC_1ST].state_dict()
                            dict_to_save['optimizer_gen_2nd'] = optimizers_dict[STEP_GEN_2ND].state_dict()
                            dict_to_save['optimizer_disc_2nd'] = optimizers_dict[STEP_DISC_2ND].state_dict()

                            torch.save(dict_to_save,
                                       os.path.join(model_base_path, 'checkpoint_%d.pth.tar' % current_step))
                            print("save model at step %d ..." % current_step)

                            for loss_key in [TTS_LOSS, ASR_LOSS]:
                                best_losses[loss_key] = curr_loss_dict[loss_key]

                        else:
                            break

            # TRAINING
            for key in modules_dict.keys():
                modules_dict[key] = modules_dict[key].train()

            # get next batch from dataloaders
            mel, spec = next(iter(wave_loader))
            T, CTC_target, CTC_len = next(iter(text_loader))
            T_for_TTS = _prepare_text_tensor(T, for_tts=True)
            T_for_ASR = _prepare_text_tensor(T, for_tts=False)

            mel, spec, T_for_TTS, T_for_ASR, CTC_target, CTC_len = \
                [_as_variable(d) for d in [mel, spec, T_for_TTS, T_for_ASR, CTC_target, CTC_len]]

            # define loss(reconstruction, GAN, discriminator)
            gen_loss = Variable(torch.FloatTensor(1).zero_())
            disc_loss = Variable(torch.FloatTensor(1).zero_())
            if args.cuda:
                gen_loss = gen_loss.cuda()
                disc_loss = disc_loss.cuda()

            loss_terms = {}

            start_time = time.time()

            if current_step < 1000:
                # discriminator pre-training step
                z_text = modules_dict[TEXT_ENC](T_for_TTS)
                mel_output, spec_output = modules_dict[WAVE_DEC](z_text, mel_input=None)
                disc_output_fake = modules_dict[W_DISC](spec_output.transpose(1, 2))
                disc_output_real = modules_dict[W_DISC](spec)
                temp_disc_loss, temp_gan_loss = discriminator_loss(disc_output_real, disc_output_fake)
                disc_loss += temp_disc_loss
                loss_terms[W_DISC_LOSS] = temp_disc_loss.data[0]

                step_key = STEP_DISC_1ST
                lr_scheds[step_key].step()
                optimizers_dict[step_key].zero_grad()
                disc_loss.backward()
                for key in modules_dict.keys():
                    nn.utils.clip_grad_norm(modules_dict[key].parameters(), 1.)
                optimizers_dict[step_key].step()
                log_counts[step_key] += 1

            elif step % 2 == 0:
                # First step
                z_text = modules_dict[TEXT_ENC](T_for_TTS)
                mel_output, spec_output = modules_dict[WAVE_DEC](z_text, mel_input=None)
                T_recon = modules_dict[ASR_MODULE](spec_output.transpose(1, 2))

                disc_output_fake = modules_dict[W_DISC](spec_output.transpose(1, 2))
                disc_output_real = modules_dict[W_DISC](spec)

                # calculate losses
                ctc_probsize = Variable(torch.IntTensor([T_recon.size()[1]] * T_recon.size()[0]))
                if args.cuda:
                    ctc_probsize = ctc_probsize.cuda()

                loss_terms[T_RECON_LOSS] = ctc_loss(T_recon, CTC_target, ctc_probsize, CTC_len, criterion_ctc, args.batch_size, args.cuda)
                loss_terms[T_RECON_LOSS] *= args.scaling_text_decoding_loss
                gen_loss += loss_terms[T_RECON_LOSS]

                temp_disc_loss, temp_gan_loss = discriminator_loss(disc_output_real, disc_output_fake)
                temp_gan_loss *= args.scaling_gan_loss
                gen_loss += temp_gan_loss
                disc_loss += temp_disc_loss

                loss_terms[T_RECON_LOSS] = loss_terms[T_RECON_LOSS].data[0]
                loss_terms[W_GAN_LOSS] = temp_gan_loss.data[0]
                loss_terms[W_DISC_LOSS] = temp_disc_loss.data[0]

                # update discriminator
                if (step / 2) % args.z_disc_step == args.z_disc_step - 1:
                    step_key = STEP_DISC_1ST
                # update generator
                else:
                    step_key = STEP_GEN_1ST

                lr_scheds[step_key].step()
                optimizers_dict[step_key].zero_grad()

                if (step / 2) % args.z_disc_step == args.z_disc_step - 1:
                    disc_loss.backward()
                else:
                    gen_loss.backward()

                for key in modules_dict.keys():
                    nn.utils.clip_grad_norm(modules_dict[key].parameters(), 1.)

                optimizers_dict[step_key].step()
                log_counts[step_key] += 1

            else:
                # Second step
                modules_dict[TEXT_ENC] = modules_dict[TEXT_ENC].train()

                # ASR_MODULE is not trained at second step
                modules_dict[ASR_MODULE] = modules_dict[ASR_MODULE].eval()

                T_output = modules_dict[ASR_MODULE](spec)

                T_decoded = _decode_as_integers(T_output)

                T_for_TTS2 = _as_variable(_prepare_text_tensor(T_decoded, for_tts=True))

                z_text2 = modules_dict[TEXT_ENC](T_for_TTS2)

                # TODO: teacher forcing is needed??
                # TODO: teacher forcing is needed??
                mel_recon, spec_recon = modules_dict[WAVE_DEC](z_text2, mel_input=mel.transpose(1, 2))
                # TODO: teacher forcing is needed??
                # TODO: teacher forcing is needed??

                # calculate losses
                loss_terms[MEL_LOSS] = mel_loss(mel_recon, mel, criterion_l1, args.cuda)
                loss_terms[SPEC_LOSS] = spec_loss(spec_recon, spec, n_priority_freq, args.cuda)
                loss_terms[W_RECON_LOSS] = loss_terms[MEL_LOSS] + loss_terms[SPEC_LOSS]
                gen_loss += loss_terms[W_RECON_LOSS]

                loss_terms[W_RECON_LOSS] = loss_terms[W_RECON_LOSS].data[0]
                loss_terms[MEL_LOSS] = loss_terms[MEL_LOSS].data[0]
                loss_terms[SPEC_LOSS] = loss_terms[SPEC_LOSS].data[0]

                step_key = STEP_GEN_2ND
                lr_scheds[step_key].step()
                optimizers_dict[step_key].zero_grad()

                gen_loss.backward()

                for key in modules_dict.keys():
                    nn.utils.clip_grad_norm(modules_dict[key].parameters(), 1.)

                optimizers_dict[step_key].step()
                log_counts[step_key] += 1

                # RECOVER MODULE STATE
                modules_dict[ASR_MODULE] = modules_dict[ASR_MODULE].train()

            time_per_step = time.time() - start_time

            # print batch logs
            for key, log_cnt in log_counts.items():
                if log_cnt == 10:
                    step_logs = '\t[STEP LOG - %s]\n' % key
                    step_logs += '\tStep: %d\n' % current_step
                    step_logs += '\tLearning rate: %.7f\n' % optimizers_dict[key].param_groups[-1]['lr']

                    for loss_key in ORDERED_LOSS_KEYS[key]:
                        step_logs += '\t%s loss: %.5f\n' % (loss_key, loss_terms[loss_key])

                    step_logs += '\t1-step time for batch size %d: %.2f\n' % (args.batch_size, time_per_step)
                    step_logs += '\t--------------------------------------'
                    print(step_logs)
                    sys.stdout.flush()

                    log_counts[key] = 0