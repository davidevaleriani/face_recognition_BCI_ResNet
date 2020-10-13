#######################################################################
# Face recognition experiment
#
#   In this experiment, the user has to find a target person in a
#   crowded real image of people walking indoor. The target is
#   represented by a young woman.
#   The user can indicate the uncertainty in his/her decision.
#
#   Command line:
#   python exp.py [-h for help]
#
#
# Author: Davide Valeriani
#         Brain-Computer Interfaces Lab
#         University of Essex
# 
#######################################################################

import sys
import pygame
import random
import numpy as N
import getpass
import time
import subprocess
import datetime
import os
import argparse
import glob
from itertools import product

target_rate = 0.25              # Percentage of trials with target
default_sessions = 6            # Number of sessions
default_repeats = 48            # Repeats per session
sessions_training = 2           # Number of sessions during training
repeats_training = 10           # Number of repetitions during training
cross_duration = 1000           # Duration of fixation cross (in ms)
stim_duration = 300             # Duration of stimuli (in ms)
mask_duration = 0               # Duration of mask (in ms)
get_uncertainty_delay = 4000    # Duration of the display to select the uncertainty
show_performance_delay = 4000   # Duration of the display showing the performance of the user (in ms)
cross_size = 100                # Size of fixation cross (in pixels)
init_uncertainty = 1.0          # Initial value of the uncertainty (0 sure, 1 unsure)
feedback = True                 # Show a feedback to the user after each session
save_results = True             # Save results in txt files

FALSE_RESP = False
TRUE_RESP = True

# Init all PyGame modules statements
pygame.init()


def reset_screen(reset_margins=True):
    # Erase screen
    screen.fill(bg_colour)
    if reset_margins:
        # Reset margins
        global margins
        margins = [0, 0, 0, 0]


# Load the image containing or not the target
def load_image(filename):
    img_name = filename.split("/")[-1][:-4]
    img = pygame.image.load(filename)
    # Scale image to fit the screen, maintaining the aspect ratio
    rect = img.get_rect()
    ratio = float(rect.width) / rect.height
    new_height = screen.get_height() - margins[0] - margins[2]
    new_width = int(ratio * new_height)
    img = pygame.transform.scale(img, (new_width, new_height))
    rect = img.get_rect()
    # Move the image in the middle of the screen
    rect.x += (screen.get_width() - rect.width - margins[1] - margins[3]) / 2
    rect.y += margins[0]
    # Return the image and the rectangle in which drawing it
    return img, rect, img_name


# Load the set of stimuli
def load_stimuli():
    stimuli = []
    is_target_present = []
    if camera == "all":
        target_images = sorted(glob.glob(stimuli_path+"*[LCR]_T*"))
        nontarget_images = sorted(glob.glob(stimuli_path+"*[LCR].*"))
        stimuli = N.hstack((N.random.choice(target_images, size=(sessions, int(target_rate*repeats)), replace=False),
                            N.random.choice(nontarget_images, size=(sessions, repeats-int(target_rate*repeats)), replace=False)))
        map(random.shuffle, stimuli)
    elif camera == "split":
        for s in session_info:
            target_images = sorted(glob.glob(stimuli_path+"%s_*%s_T*" % s))
            nontarget_images = sorted(glob.glob(stimuli_path+"%s_*%s.*" % s))
            session_images = target_images[:int(target_rate*repeats)]+nontarget_images[:(repeats-int(target_rate*repeats))]
            random.shuffle(session_images)
            stimuli.append(session_images)
    else:
        target_images = sorted(glob.glob(stimuli_path+"*"+camera+"_T*"))
        nontarget_images = sorted(glob.glob(stimuli_path+"*"+camera+".*"))
        stimuli = N.hstack((N.random.choice(target_images, size=(sessions, int(target_rate*repeats)), replace=False),
                            N.random.choice(nontarget_images, size=(sessions, repeats-int(target_rate*repeats)), replace=False)))
        map(random.shuffle, stimuli)

    for s in range(len(session_info)):
        is_target_present.append([True if "_T" in fn else False for fn in stimuli[s]])

    return stimuli, is_target_present


# Draw a fixation cross in the center of the screen
def draw_cross():
    reset_screen()
    pygame.draw.line(screen,
                     cross_colour,
                     ((screen.get_width() - cross_size) / 2, screen.get_height() / 2),
                     ((screen.get_width() + cross_size) / 2, screen.get_height() / 2),
                     4)
    pygame.draw.line(screen,
                     cross_colour,
                     (screen.get_width() / 2, (screen.get_height() - cross_size) / 2),
                     (screen.get_width() / 2, (screen.get_height() + cross_size) / 2),
                     4)
    pygame.display.flip()


# Draw a mask in the center of the screen
def draw_mask():
    reset_screen(False)
    # loading the image
    texture = pygame.image.load("../textures/texture.jpg")
    # reduce size to the drawable area
    texture = pygame.transform.scale(texture, (screen.get_width() - margins[1] - margins[3], screen.get_height() - margins[0] - margins[2]))
    texture_rect = texture.get_rect()
    # align to the middle
    texture_rect.x = margins[1]
    texture_rect.y = margins[0]
    # draw the mask
    screen.blit(texture, texture_rect)
    pygame.display.flip()


# Show a message and wait for the left mouse button
def show_message_and_wait(text, colour=(255, 255, 0), wait=True, show_target=None):
    reset_screen()
    my_font = pygame.font.SysFont("Arial", 40)
    text_split = text.split("\n")
    num_of_lines = len(text_split)
    tot_offset_y = (num_of_lines-1) * 40
    if show_target:
        tot_offset_y += 200
    for sentence in text_split:
        label = my_font.render(sentence, 1, colour)
        offset_y = text_split.index(sentence) * (my_font.size(sentence)[1] + 40)
        screen.blit(label, ((screen.get_width()-my_font.size(sentence)[0])/2,
                            ((screen.get_height()-tot_offset_y-my_font.size(sentence)[1]*num_of_lines)/2+offset_y)))
    if show_target:
        texture = pygame.image.load("images/target_"+show_target+".pgm")
        texture = pygame.transform.scale(texture, (200, 200))
        texture_rect = texture.get_rect()
        texture_rect.x = (screen.get_width()-200)/2
        texture_rect.y = screen.get_height()/2+tot_offset_y-200
        screen.blit(texture, texture_rect)
    pygame.display.flip()
    if wait:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_pressed()[0]:
                    return


# Run a new session
def run_experiment(stimulus):
    # Show the fixation cross for cross_duration milliseconds
    draw_cross()
    pygame.time.delay(cross_duration)
    # Ignore button pressed during the fixation cross
    pygame.event.clear()
    reset_screen()
    # Show the stimuli for stim_duration milliseconds
    img_surface, img_rect, image_name = load_image(stimulus)
    screen.blit(img_surface, img_rect)
    # Update the margins to show the mask of the same size of the image
    global margins
    instructions_margin = margins[0]
    margins[0] = instructions_margin + (screen.get_height() - img_rect.h - instructions_margin) / 2
    margins[2] = (screen.get_height() - img_rect.h - instructions_margin) / 2
    margins[1] = margins[3] = (screen.get_width() - img_rect.w) / 2
    header = 'T='+str(is_target_present[current_session][repeat])+',IM='+image_name
    bsusb.send(header, code=bsusb.CTL)  # 0 = stimulus marker
    pygame.event.clear()
    clock.tick()
    pygame.display.flip()
    pygame.time.delay(stim_duration)
    # Show the mask for mask_duration milliseconds
    if mask_duration > 0:
        draw_mask()
        pygame.time.delay(mask_duration)
    reset_screen()
    pygame.display.flip()
    return image_name


# Ask the user to insert his/her uncertainty in the current decision
def ask_uncertainty(resp):
    progress_step = 0.1
    bar_height = 150
    bar_position_top = (screen.get_height()-bar_height) / 2
    side_margin = 100
    uncertainty = init_uncertainty  # 1 = Unsure, 0 = Sure
    # Set a timeout after which the uncertainty value is collected
    timeout = get_uncertainty_delay  # ms
    timeout_clock = pygame.time.Clock()
    timeout_clock.tick()
    while timeout > 0:
        reset_screen()
        event = pygame.event.get(pygame.MOUSEBUTTONDOWN)
        if event and event[0].type == pygame.MOUSEBUTTONDOWN and event[0].button == 5 and uncertainty < 1.0:
            uncertainty += progress_step
            # Send the event to USB to keep track of the decision-making process of the user
            bsusb.send("ConfDown_"+str(uncertainty), code=bsusb.CTL)
        elif event and event[0].type == pygame.MOUSEBUTTONDOWN and event[0].button == 4 and uncertainty > 0.0:
            uncertainty -= progress_step
            # Send the event to USB to keep track of the decision-making process of the user
            bsusb.send("ConfUp_"+str(uncertainty), code=bsusb.CTL)
        uncertainty = round(uncertainty, 1)
        text_right = "Very confident"
        text_left = "Not confident"
        # The bar is an horizontal histogram of the same colour of the response
        bar_length = (1-uncertainty) * (screen.get_width() - 2*side_margin)
        bar_position_left = side_margin
        # Black/White bar
        # bar_colour = ((1-uncertainty)*255, (1-uncertainty)*255, (1-uncertainty)*255)

        # Bar of different colours depending from the response selected
        if uncertainty == 1.0:
            bar_colour = (25, 25, 25)
        elif resp == TRUE_RESP:
            # Green(ish)
            bar_colour = (0, 100+(1-uncertainty)*155, 0)
        else:
            # Cyan(ish)
            bar_colour = (0, (1-uncertainty) * 255, 255)
        pygame.draw.rect(screen, bar_colour, pygame.Rect(bar_position_left, bar_position_top, bar_length, bar_height))
        # Add text on the left side
        myfont = pygame.font.SysFont("Arial", 40)
        label = myfont.render(text_left, 1, colours["white"])
        screen.blit(label, (side_margin, (screen.get_height()-bar_height-2*myfont.size(text_left)[1])/2))
        # Add text on the right side
        label = myfont.render(text_right, 1, colours["white"])
        screen.blit(label, (screen.get_width()-side_margin-myfont.size(text_right)[0], (screen.get_height()-bar_height-2*myfont.size(text_right)[1])/2))
        # Update the screen
        pygame.display.flip()
        timeout -= timeout_clock.tick()
    return uncertainty


def wait_for_response(target_type=None):
    """
    Get the response and response times from the subject.
    :return: response, response time
    """
    reset_screen()
    show_message_and_wait("Have you seen the target?\nYES / NO", wait=False, colour=colours["white"],
                          show_target=target_type)
    pygame.event.clear()
    clock.tick()
    sys.stdout.flush()
    while True:
        if pygame.event.get(pygame.MOUSEBUTTONDOWN):
            # Get the response
            if pygame.mouse.get_pressed()[0]:
                r = TRUE_RESP
            elif pygame.mouse.get_pressed()[2]:
                r = FALSE_RESP
            else:
                continue
            # Get elapsed time
            clock.tick()
            t = clock.get_time()/1000.0
            return r, t


# Print settings of the run
def print_settings():
    os.system('tput reset')
    print '*'*60
    print '*', ' '*23, "SETTINGS", ' '*23, '*'
    print '*'*60
    print " Date:\t\t\t", str(datetime.datetime.now())[:-7]
    print " Subject ID:\t\t", subject_ID
    print " Sessions:\t\t", sessions
    print " Repetitions:\t\t", repeats
    print " Mode:\t\t\t", "Testing" if not training_session else "Training"
    print '*'*60
    print


# Show the instructions of the experiment and the target face on the screen
def show_instructions():
    reset_screen()
    title = "INSTRUCTIONS"
    title_font = pygame.font.SysFont("Arial", 40)
    text = title_font.render(title, True, colours["white"])
    screen.blit(text, ((screen.get_width()-title_font.size(title)[0])/2, 20))
    main_font = pygame.font.SysFont("Arial", 27)
    instructions = """You are going to undertake a face recognition experiment in which you have
    to identify a target person in a crowded scene. The experiment is split
    in %s sessions of %s trials each.
In each session, you will be assigned a target person. A picture of the
    target face is shown at the beginning of the session. Please memorise it.
    The target person could vary across different sessions but it is always
    the same within the same session.
In each trial of the session, you will see a fixation cross followed by
    a picture of a crowded scene recorded from a security camera. Your task
    is to decide whether the target person was present or not in that scene.
    Please note that the scene will be shown for a very limited time,
    making the task quite challenging.
After each scene you will be asked to indicate your decision as quickly
    as possible by pressing the left mouse button to say that you have seen
    the target person or the right mouse button to say that the target was
    not present (LEFT=YES     RIGHT=NO). Then, you will have to indicate
    how confident you were on that decision using the mouse wheel
    (scroll UP/DOWN = MORE/LESS confident).

Press the left mouse button when you are ready to start.
""" % (str(sessions), str(repeats))
    text_split = instructions.split("\n")
    sizes_sentences = [main_font.size(s)[0] for s in text_split]
    left_margin = (screen.get_width()-max(sizes_sentences))/2
    offset_y = title_font.size(title)[1] + 20
    for sentence in text_split:
        label = main_font.render(sentence, 1, colours["white"])
        offset_y += main_font.size(sentence)[1]+1
        screen.blit(label, (left_margin, offset_y))
    pygame.display.flip()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_pressed()[0]:
                return


########
# MAIN
########

# Get parameters from command line
parser = argparse.ArgumentParser(description="Target Detection - Experiment 1")
parser.add_argument("-t", "--training", help="Perform a training session (disable USB and logs)", default=False, action="store_true")
parser.add_argument("-s", "--sessions", help="Number of sessions", default=default_sessions, type=int)
parser.add_argument("-r", "--repetitions", help="Number of repetitions per session", default=default_repeats, type=int)
parser.add_argument("-f", "--fullscreen", help="Show the experiment full screen", default=False, action="store_true")
parser.add_argument("-c", "--camera", help="Which camera should be used (default = split)", default="split", choices=["L", "C", "R", "all", "split"])
parser.add_argument("-q", "--quality", help="Type of stimuli (default = bw)", default="bw", choices=["colour", "bw"])
args = parser.parse_args()

training_session = args.training
sessions = args.sessions
repeats = args.repetitions
fullscreen = args.fullscreen
camera = args.camera
quality = args.quality

if not training_session:
    # Online mode: send data to USB
    # Ensure USB is properly initialised
    username = getpass.getuser()
    subprocess.call(["gksudo", "chown -R %s /dev/bus/usb/" % username])
    sys.path.append('../../../BCI-Mouse/sources/BioSemiUSB/out/')
    import bsusb
else:
    # Offline mode: store USB data in a file
    sys.path.append('../../../BCI-Mouse/sources/BioSemiUSB/src/')
    import bsusb2file as bsusb
    save_results = False
    sessions = sessions_training
    repeats = repeats_training

session_info = list(product(["bg1", "bg2"], ["L", "C", "R"]))
bsusb.init()

if quality == "colour":
   stimuli_path = "images/"
elif quality == "bw":
   stimuli_path = "imagesBW/"

if save_results:
    # Get subject number
    subject_ID = raw_input("Subject ID (e.g., 001, 002,...):")
    # Get the age of the participant
    while True:
        age = raw_input("Age: ")
        try:
            if 18 <= int(age) <= 50:
                break
        except:
            continue
    while True:
        gender = raw_input("Gender (M/F): ").upper()
        if gender in ["M", "F"]:
            break
    while True:
        hand = raw_input("Preferred hand (L/R): ").upper()
        if hand in ["L", "R"]:
            break
    experiment_date = datetime.date.today().strftime("%d/%m/%Y")
    logfile = open('Subj_%s_stimuli.txt' % subject_ID, 'w')
    info = open('Subj_%s_info.txt' % subject_ID, 'w')
    info.write(age+","+gender+","+hand+","+experiment_date+"\n")
    info.close()
    errors_file = open('Subj_%s_errors.txt' % subject_ID, 'w')
    errors_file.write(str(sessions)+" "+str(repeats)+"\n")
    print_settings()

# Randomise sequence of sessions (different for each subject) but NOT the stimuli (same random order for each subject)
session_index = range(len(session_info))
random.shuffle(session_index)
session_index = session_index[:sessions]

# Same order of stimuli for each subject in each session, but different from training and practicing
if not training_session:
    random.seed(100)
else:
    random.seed(200)

# Generate list of stimuli
stimuli, is_target_present = load_stimuli()

# Save the order of the session in the log file
if save_results:
    logfile.write(" ".join(map(str, session_index))+"\n")
# Set colours
colours = {'white': (255, 255, 255),
           'black': (0, 0, 0),
           'yellow': (255, 255, 0),
           'gray': (128, 128, 128),
           }
bg_colour = colours["black"]
cross_colour = colours["gray"]
# Results variables
clock = pygame.time.Clock()
resp = []
times = []
# Create a full screen window
if fullscreen:
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
else:
    screen = pygame.display.set_mode((1024, 768))
pygame.display.set_caption("Target detection")
# Hide the mouse cursor
pygame.mouse.set_visible(False)

# Build the list of sessions with the target
for s in range(sessions):
    resp.append([FALSE_RESP for _ in range(repeats)])
    times.append([0.0 for _ in range(repeats)])

show_instructions()
score_sessions = []

for num_session, current_session in enumerate(session_index):
    target_type = "bg1" if "bg1" in stimuli[current_session][0] else "bg2"
    show_message_and_wait("Session # "+str(num_session+1)+'\n'+"The target of this session is shown below\n"+
                          "Press YES to start", colour=colours["white"], show_target=target_type)
    bsusb.send('FaceRecognition_targetRate='+str(target_rate)+
               '_session='+str(current_session), code=bsusb.SOR)
    for repeat, stimulus in enumerate(stimuli[current_session]):
        image_name = run_experiment(stimulus)
        # Wait for the response from the user
        resp[num_session][repeat], times[num_session][repeat] = wait_for_response(target_type)
        text = 'correct' if resp[num_session][repeat] == is_target_present[current_session][repeat] else "incorrect"
        # Send the response to USB and write on the logfile response and uncertainty
        bsusb.send(text, code=bsusb.CTL + bsusb.TRG)  # response marker
        # Ask the uncertainty using a bar
        uncertainty = ask_uncertainty(resp[num_session][repeat])
        # Send the final uncertainty to USB
        bsusb.send("Conf="+str(uncertainty), code=bsusb.CTL)
        if save_results:
            header = 'T='+str(is_target_present[current_session][repeat])+',IM='+str(image_name)
            logfile.write(header+","+str(uncertainty)+","+text+","+str(times[num_session][repeat])+"\n")
            logfile.flush()
    pygame.time.delay(250)
    errors_current_session = N.sum(resp[num_session][i] != is_target_present[current_session][i] for i in range(len(resp[num_session])))
    if save_results:
        errors_file.write(str(num_session)+" "+str(errors_current_session)+"\n")
    time.sleep(0.2)
    bsusb.send('END', bsusb.EOR)
    time.sleep(0.2)
    score_sessions.append(((repeats - errors_current_session) / float(repeats)) * 100)
    if feedback:
        show_message_and_wait("Your score in the last session:"+'\n'+
                              str(score_sessions[-1])+" %",
                              colours["white"], wait=False)
        pygame.time.delay(show_performance_delay)

show_message_and_wait("Thank you for your participation"+'\n'+"Press YES to end", colour=colours["white"])

# Print stats
print
print "Ground truth:\t", N.asarray(is_target_present)[session_index].tolist()
print "Responses:\t", resp
print "Response times:\t", times
print "Average RT:\t", round(N.mean(times), 3), "s"
print "-"*40
print "Correct ratio:\t\t %.2f %%" % N.mean(score_sessions)
print

if save_results:
    logfile.close()
    errors_file.close()
