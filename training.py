import sys
import time
from itertools import chain

import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

from data import *
from etc import *
from network import *
import paths

parser = argparse.ArgumentParser(description='Unsupervised Voice Conversion')

# dataset params
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
parser.add_argument('--disc_step', type=int, default=3)
parser.add_argument('--gan_loss_ratio', type=float, default=1.)

# loss arguments
parser.add_argument('--feature_recon_loss', type=str2bool, help='bottleneck feature reconstruction loss', default=True)
parser.add_argument('--feature_matching_loss', type=str2bool, help='feature matching loss', default=False)

# etc
parser.add_argument('--cuda', type=str2bool, help='whether to use cuda(GPU)', default=True)
parser.add_argument('--save_step', type=int, help='steps btw model and image', default=5000)
parser.add_argument('--save_result', type=str2bool, default=True)

parser.add_argument('--model_name', type=str, default='unsupervised_vc')
parser.add_argument('--restore_step', type=int, help='Global step to restore checkpoint', default=0)


# TODO: implement feature matching loss
def discriminator_loss(d_real, d_fake):
    d_loss = 0.5 * (torch.mean((d_real - 1) ** 2) + torch.mean(d_fake ** 2))
    g_loss = 0.5 * torch.mean((d_fake - 1) ** 2)
    return d_loss, g_loss


def spec_loss(s_synth, s_target, priority_freq, cuda=True):
    min_length = min(s_synth.size()[1], s_target.size()[1])
    l1loss = torch.abs(s_synth[:, :min_length, :] - s_target[:, :min_length, :])
    l1loss_with_priority = 0.5 * torch.mean(l1loss) + 0.5 * torch.mean(l1loss[:, :priority_freq, :])
    if cuda:
        l1loss_with_priority = l1loss_with_priority.cuda()
    return l1loss_with_priority


def feature_reconstruction_loss(f_synth, f_source, cuda=True):
    l1loss = torch.mean(torch.abs(f_synth - f_source))
    if cuda:
        l1loss = l1loss.cuda()
    return l1loss


def feature_matching_loss(f_synth, f_recon, cuda=True):
    l1loss = Variable(torch.FloatTensor(1).zero_())
    if cuda:
        l1loss = l1loss.cuda()
    for f1, f2 in zip(f_synth, f_recon):
        l1loss += torch.mean(torch.abs(f1 - f2))
    return l1loss


if __name__ == '__main__':
    args = parser.parse_args()

    dataset_A_tr_path = paths.DATA_BLIZZARD_TRAINING_PATH
    dataset_A_te_path = paths.DATA_BLIZZARD_TEST_PATH

    dataset_B_tr_path = paths.DATA_LJ_TRAINING_PATH
    dataset_B_te_path = paths.DATA_LJ_TEST_PATH

    model_base_path = os.path.join(paths.MODEL_BASE_PATH, args.model_name)
    if not os.path.exists(model_base_path):
        os.mkdir(model_base_path)

    result_base_path = os.path.join(paths.RESULT_PATH, args.model_name)
    tr_ev_result_path = os.path.join(result_base_path, paths.TRAINING_STR)
    te_ev_result_path = os.path.join(result_base_path, paths.TEST_STR)
    for result_path in [result_base_path, tr_ev_result_path, te_ev_result_path]:
        if not os.path.exists(result_path):
            os.mkdir(result_path)

    # Get dataset
    wave_A_tr = AudioDataset(dataset_A_tr_path, args)
    wave_B_tr = AudioDataset(dataset_B_tr_path, args)
    wave_A_te = AudioDataset(dataset_A_te_path, args)
    wave_B_te = AudioDataset(dataset_B_te_path, args)

    # Construct model
    modules_dict = {ENC_A: BottleneckEncoder(),
                    ENC_B: BottleneckEncoder(),
                    DEC_A: BottleneckDecoder(),
                    DEC_B: BottleneckDecoder(),
                    DISC_A: Discriminator(),
                    DISC_B: Discriminator()}

    if args.cuda:
        for key, module in modules_dict.items():
            modules_dict[key] = torch.nn.DataParallel(module).cuda()

    # Make optimizer
    generator_optimizer = optim.Adam(chain.from_iterable(
        [modules_dict[module].parameters() for module in [ENC_A, ENC_B, DEC_A, DEC_B]]), lr=args.lr_gen)
    discriminator_optimizer = optim.Adam(chain.from_iterable(
        [modules_dict[module].parameters() for module in [DISC_A, DISC_B]]), lr=args.lr_disc)

    # Load checkpoint if exists
    try:
        print('loading checkpoint from ' + model_base_path)
        checkpoint = torch.load(os.path.join(model_base_path, 'checkpoint_%d.pth.tar' % args.restore_step))
        for key in modules_dict.keys():
            modules_dict[key].load_state_dict(checkpoint['%s_state_dict' % key])

        generator_optimizer = optim.Adam(chain.from_iterable(
            [modules_dict[module].parameters() for module in [ENC_A, ENC_B, DEC_A, DEC_B]]), lr=args.lr_gen)
        discriminator_optimizer = optim.Adam(chain.from_iterable(
            [modules_dict[module].parameters() for module in [DISC_A, DISC_B]]), lr=args.lr_disc)
        generator_optimizer.load_state_dict(checkpoint['generator_optimizer'])
        discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])
        for opt in [generator_optimizer, discriminator_optimizer]:
            opt.param_groups[-1]['initial_lr'] = opt.param_groups[-1]['lr']
        print("\n--------model restored at step %d--------\n" % args.restore_step_tts)
    except:
        print("\n--------Start New Training--------\n")

    print_modules_size(modules_dict)

    # define parameters to be updated, optimizers to be used and schedulers for learning rate control
    generator_scheduler = lr_scheduler.StepLR(generator_optimizer, step_size=args.decay_step, gamma=args.decay_rate)
    discriminator_scheduler = lr_scheduler.StepLR(discriminator_optimizer, step_size=args.decay_step, gamma=args.decay_rate)

    # Loss for frequency of human register
    # TODO: ablation test needed
    n_priority_freq = int(3000 / (args.sr * 0.5) * args.num_freq)

    for epoch in range(args.epochs):
        # create shuffled dataloaders
        wave_loader_A = DataLoader(wave_A_tr, batch_size=args.batch_size, shuffle=True,
                                   drop_last=True, collate_fn=collate_fn_audio, num_workers=8)
        wave_loader_B = DataLoader(wave_B_tr, batch_size=args.batch_size, shuffle=True,
                                   drop_last=True, collate_fn=collate_fn_audio, num_workers=8)

        dataloader_length = min(len(wave_loader_A), len(wave_loader_B))
        for step in range(dataloader_length):
            current_step = step + args.restore_step + epoch * dataloader_length + 1

            # EVALUATION, parameters and result saving
            if current_step % args.save_step == 0:
                for key in modules_dict.keys():
                    modules_dict[key] = modules_dict[key].eval()

                for phase_str, dataset_A, dataset_B, result_path in zip(["TRAINING", "TEST"],
                                                                        [wave_A_tr, wave_A_te],
                                                                        [wave_B_tr, wave_B_tr],
                                                                        [tr_ev_result_path, te_ev_result_path]):

                    eval_data_loader_A = DataLoader(dataset_A, batch_size=args.batch_size, shuffle=False,
                                                    drop_last=False, collate_fn=collate_fn_audio, num_workers=8)
                    eval_data_loader_B = DataLoader(dataset_B, batch_size=args.batch_size, shuffle=False,
                                                    drop_last=False, collate_fn=collate_fn_audio, num_workers=8)

                    # TODO: ablation test needed for recon_loss and fm_loss
                    curr_loss_dict = {key: 0. for key in [RECON_LOSS_A, RECON_LOSS_B,
                                                          CYCLE_LOSS_A, CYCLE_LOSS_B,
                                                          FM_LOSS_A, FM_LOSS_B,
                                                          FR_LOSS_A, FR_LOSS_B,
                                                          GAN_LOSS_A, GAN_LOSS_B,
                                                          DISC_LOSS_A, DISC_LOSS_B]}

                    eval_dataloader_length = min(len(eval_data_loader_A), len(eval_data_loader_B)) - 1
                    for j in range(eval_dataloader_length):
                        if phase_str == "TRAINING" and j >= 50:
                            break

                        # get next batch from dataloader
                        A = Variable(next(iter(eval_data_loader_A)))
                        B = Variable(next(iter(eval_data_loader_B)))
                        if args.cuda:
                            A = A.cuda()
                            B = B.cuda()
                        A = A.unsqueeze(1)
                        B = B.unsqueeze(1)

                        # Forward
                        z_A, f_enc_A = modules_dict[ENC_A](A)
                        A_rec, f_dec_A_rec = modules_dict[DEC_A](z_A)
                        AB, f_dec_AB = modules_dict[DEC_B](z_A)
                        z_AB, f_enc_AB = modules_dict[ENC_B](AB)
                        ABA, f_dec_ABA = modules_dict[DEC_A](z_AB)

                        z_B, f_enc_B = modules_dict[ENC_B](B)
                        B_rec, f_dec_B_rec = modules_dict[DEC_B](z_B)
                        BA, f_dec_BA = modules_dict[DEC_A](z_B)
                        z_BA, f_enc_BA = modules_dict[ENC_A](BA)
                        BAB, f_dec_BAB = modules_dict[DEC_B](z_BA)

                        # TODO: use reconstructed spectrograms(A_rec, B_rec, ABA, BAB) while training DISC, GAN loss?
                        disc_A = modules_dict[DISC_A](A)
                        disc_AB = modules_dict[DISC_B](AB)
                        # disc_A_rec = modules_dict[DISC_A](A_rec)
                        # disc_ABA = modules_dict[DISC_A](ABA)

                        disc_B = modules_dict[DISC_B](B)
                        disc_BA = modules_dict[DISC_A](BA)
                        # disc_B_rec = modules_dict[DISC_B](B_rec)
                        # disc_BAB = modules_dict[DISC_B](BAB)

                        # calculate loss
                        curr_loss_dict[RECON_LOSS_A] += spec_loss(A_rec, A, n_priority_freq, args.cuda).data[0]
                        curr_loss_dict[RECON_LOSS_B] += spec_loss(B_rec, B, n_priority_freq, args.cuda).data[0]
                        curr_loss_dict[CYCLE_LOSS_A] += spec_loss(ABA, A, n_priority_freq, args.cuda).data[0]
                        curr_loss_dict[CYCLE_LOSS_B] += spec_loss(BAB, B, n_priority_freq, args.cuda).data[0]
                        curr_loss_dict[FM_LOSS_A] += feature_matching_loss(f_dec_ABA, f_enc_A, args.cuda).data[0]
                        curr_loss_dict[FM_LOSS_B] += feature_matching_loss(f_dec_BAB, f_enc_B, args.cuda).data[0]
                        curr_loss_dict[FR_LOSS_A] += feature_reconstruction_loss(z_A, z_AB, args.cuda).data[0]
                        curr_loss_dict[FR_LOSS_B] += feature_reconstruction_loss(z_B, z_BA, args.cuda).data[0]
                        temp_disc_A, temp_gan_A = discriminator_loss(disc_A, disc_BA)
                        temp_disc_B, temp_gan_B = discriminator_loss(disc_B, disc_AB)
                        curr_loss_dict[GAN_LOSS_A] += temp_gan_A.data[0]
                        curr_loss_dict[GAN_LOSS_B] += temp_gan_B.data[0]
                        curr_loss_dict[DISC_LOSS_A] += temp_disc_A.data[0]
                        curr_loss_dict[DISC_LOSS_B] += temp_disc_B.data[0]

                        # save result files
                        if j == 0:
                            result_epoch_path = os.path.join(result_path, str(current_step))
                            if not os.path.exists(result_epoch_path):
                                os.mkdir(result_epoch_path)

                            audio_writer(result_epoch_path, A_rec, name_head='A_rec', use_cuda=args.cuda)
                            audio_writer(result_epoch_path, BA, name_head='BA', use_cuda=args.cuda)
                            audio_writer(result_epoch_path, ABA, name_head='ABA', use_cuda=args.cuda)
                            audio_writer(result_epoch_path, B_rec, name_head='B_rec', use_cuda=args.cuda)
                            audio_writer(result_epoch_path, AB, name_head='AB', use_cuda=args.cuda)
                            audio_writer(result_epoch_path, BAB, name_head='BAB', use_cuda=args.cuda)

                    step_logs = '[TEST LOG - %s]\n' % phase_str
                    step_logs += 'Step: %d\n' % current_step

                    for loss_key in curr_loss_dict.keys():
                        step_logs += '%s loss: %.5f\n' % (loss_key, curr_loss_dict[loss_key] / eval_dataloader_length)

                    step_logs += '--------------------------------------'
                    print(step_logs)
                    sys.stdout.flush()

                # save model
                dict_to_save = {}
                for key in modules_dict.keys():
                    dict_to_save['%s_state_dict' % key] = modules_dict[key].state_dict()
                    dict_to_save['generator_optimizer'] = generator_optimizer.state_dict()
                    dict_to_save['discriminator_optimizer'] = discriminator_optimizer.state_dict()
                    torch.save(dict_to_save, os.path.join(model_base_path, 'checkpoint_%d.pth.tar' % current_step))
                    print("save model at step %d ..." % current_step)

            # TRAINING
            for key in modules_dict.keys():
                modules_dict[key] = modules_dict[key].train()

            start_time = time.time()
            loss_dict = {}

            # get next batch from dataloader
            A = Variable(next(iter(wave_loader_A)))
            B = Variable(next(iter(wave_loader_A)))
            if args.cuda:
                A = A.cuda()
                B = B.cuda()
            A = A.unsqueeze(1)
            B = B.unsqueeze(1)

            # Forward
            z_A, f_enc_A = modules_dict[ENC_A](A)
            A_rec, f_dec_A_rec = modules_dict[DEC_A](z_A)
            AB, f_dec_AB = modules_dict[DEC_B](z_A)
            z_AB, f_enc_AB = modules_dict[ENC_B](AB)
            ABA, f_dec_ABA = modules_dict[DEC_A](z_AB)

            z_B, f_enc_B = modules_dict[ENC_B](B)
            B_rec, f_dec_B_rec = modules_dict[DEC_B](z_B)
            BA, f_dec_BA = modules_dict[DEC_A](z_B)
            z_BA, f_enc_BA = modules_dict[ENC_A](BA)
            BAB, f_dec_BAB = modules_dict[DEC_B](z_BA)

            # TODO: use reconstructed spectrograms(A_rec, B_rec, ABA, BAB) while training DISC, GAN loss?
            disc_A = modules_dict[DISC_A](A)
            disc_AB = modules_dict[DISC_B](AB)
            # disc_A_rec = modules_dict[DISC_A](A_rec)
            # disc_ABA = modules_dict[DISC_A](ABA)

            disc_B = modules_dict[DISC_B](B)
            disc_BA = modules_dict[DISC_A](BA)
            # disc_B_rec = modules_dict[DISC_B](B_rec)
            # disc_BAB = modules_dict[DISC_B](BAB)

            loss_dict[RECON_LOSS_A] = spec_loss(A_rec, A, n_priority_freq, args.cuda)
            loss_dict[RECON_LOSS_B] = spec_loss(B_rec, B, n_priority_freq, args.cuda)
            loss_dict[CYCLE_LOSS_A] = spec_loss(ABA, A, n_priority_freq, args.cuda)
            loss_dict[CYCLE_LOSS_B] = spec_loss(BAB, B, n_priority_freq, args.cuda)
            loss_dict[DISC_LOSS_A], loss_dict[GAN_LOSS_A] = discriminator_loss(disc_A, disc_BA)
            loss_dict[DISC_LOSS_B], loss_dict[GAN_LOSS_B] = discriminator_loss(disc_B, disc_AB)
            loss_dict[FM_LOSS_A] = feature_matching_loss(f_dec_ABA, f_enc_A, args.cuda)
            loss_dict[FM_LOSS_B] = feature_matching_loss(f_dec_BAB, f_enc_B, args.cuda)
            loss_dict[FR_LOSS_A] = feature_reconstruction_loss(z_A, z_AB, args.cuda)
            loss_dict[FR_LOSS_B] = feature_reconstruction_loss(z_B, z_BA, args.cuda)

            if current_step % args.disc_step == 0:
                discriminator_optimizer.zero_grad()
                (loss_dict[DISC_LOSS_A] + loss_dict[DISC_LOSS_B]).backward()

                # gradient clipping
                for key in modules_dict.keys():
                    nn.utils.clip_grad_norm(modules_dict[key].parameters(), 1.)

                discriminator_optimizer.step()
                discriminator_scheduler.step()

            else:
                generator_optimizer.zero_grad()
                # TODO: implement and test curriculum learning
                gen_loss = loss_dict[RECON_LOSS_A] + loss_dict[RECON_LOSS_B] + \
                           loss_dict[CYCLE_LOSS_A] + loss_dict[CYCLE_LOSS_B] + \
                           (loss_dict[GAN_LOSS_A] + loss_dict[GAN_LOSS_B]) * args.gan_loss_ratio
                if args.feature_recon_loss:
                    gen_loss += loss_dict[FR_LOSS_A] + loss_dict[FR_LOSS_B]
                if args.feature_matching_loss:
                    gen_loss += loss_dict[FM_LOSS_A] + loss_dict[FM_LOSS_B]

                # gradient clipping
                for key in modules_dict.keys():
                    nn.utils.clip_grad_norm(modules_dict[key].parameters(), 1.)

                generator_optimizer.step()
                generator_scheduler.step()

            time_per_step = time.time() - start_time

            if current_step % 10:
                step_logs = '\t[TRAINING LOG]\n'
                step_logs += '\tStep: %d\n' % current_step
                for loss in [RECON_LOSS_A, RECON_LOSS_B, CYCLE_LOSS_A, CYCLE_LOSS_B,
                             FM_LOSS_A, FM_LOSS_B, FR_LOSS_A, FR_LOSS_B,
                             GAN_LOSS_A, GAN_LOSS_B, DISC_LOSS_A, DISC_LOSS_B]:
                    step_logs += '\t%s loss: %.5f\n' % (loss, loss_dict[loss])
                step_logs += '\tTraining time per step with batch size %d: %.2f\n' % (args.batch_size, time_per_step)
                step_logs += '\t--------------------------------------'
                print(step_logs)
                sys.stdout.flush()
