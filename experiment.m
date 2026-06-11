% MATLAB EEG Experiment - 4 blocks (1 practice) with LSL Markers.
% --- 1. Experiment Setup ---
clear;
close all;
clc;
% Add Paths
addpath(genpath('C:\Users\cogexp\Downloads\3.0.19.14\Psychtoolbox')); %Psychtoolbox
addpath(genpath('C:\Users\cogexp\Desktop\liblsl-Matlab-master')); %LSL
% Experiment params
numBlocks = 4;
% MODIFICATION: Adjusted trialsPerBlock for practice block (15 trials per phase * 3 phases = 45)
%trialsPerBlock for blocks 2,3,4 should be 148 to ensure 2 repetitions per word, but block 1 (practice) has 45 trials.
trialsPerBlock = 148; % This will be overridden for Block 1 only.
practiceTrialsPerPhase = 15; % practice block
stimulusDuration = 3.0;
itiDuration = 1.5; % in s
breakDuration = 30; % in s (unused, key press controls progression)
fixationCrossDuration = 3.5; % block 4 only

% Text stimuli
allStimulitext = {
        'Hello!',
        'Yes!',
        'No.',
        'Water',
        'Tea',
        'Coffee',
        'Food',
        'Up',
        'Down',
        'Left',
        'Right',
        'Light',
        'Dark',
        'Head',
        'Arm',
        'Arms',
        'Leg',
        'Legs',
        'Heart',
        'Hands',
        'Foot',
        'Feet',
        'Zero',
        'One',
        'Two',
        'Three',
        'Four',
        'Five',
        'Six',
        'Seven',
        'Eight',
        'Nine',
        'Times',
        'Stop!',
        'Go on.',
        'Before',
        'Now.',
        'After',
        'Yesterday',
        'Today',
        'Tomorrow',
        'Shower',
        'Wash',
        'Bath',
        'Love',
        'Daughter',
        'Son',
        'Partner',
        'Assistant',
        'Nurse',
        'Music',
        'Drink',
        'Jam',
        'Yellow',
        'What?',
        'Where?',
        'When?',
        'Why?',
        'Who?',
        'How?',
        'Thank you',
        'I need',
        'I want',
        'I can',
        'I can t',
        'Sure!',
        'Joy',
        'Help!',
        'Quit',
        'Hot',
        'Cold',
        'Egg',
        'Ribbon',
        'Palm',
};
numStimuli = length(allStimulitext);

% --- LSL Configuration ---
% Initializing LSL library and setting up LSL marker stream
disp('Loading LSL library...');
lib = lsl_loadlib();
disp('Creating LSL marker stream...')
info = lsl_streaminfo(lib, 'VeronicaExperimentMarkers', 'Markers', 1, 0, 'cf_int16');
%info = lsl_streaminfo(lib, 'MyExperimentMarkers', 'Markers', 1, 0, 'cf_int32', 'myexp_uid123');
outlet = lsl_outlet(info);
disp('LSL marker stream ready to  use!');
% --- Data Storage Setup ---
% Columns: Block, Trial, StimulusType (index), Timestamp, Presentedtext, ButtonPressTimestamp
experimentLog = cell(numBlocks * trialsPerBlock, 6);
logIndex = 1;

% --- Psychtoolbox Setup ---
try
    % uncomment to skip Psychtoolbox sync tests
    % Screen('Preference', 'SkipSyncTests', 1);
    screens = Screen('Screens');
    screenNumber = max(screens);
    % Define colors (RGB 0-255)
    white = WhiteIndex(screenNumber);
    black = BlackIndex(screenNumber);
    grey = white / 2;
    [window, windowRect] = PsychImaging('OpenWindow', screenNumber, grey);
    % Get the size of the window (in pixels) to center
    [xCenter, yCenter] = RectCenter(windowRect);
    %text properties
    Screen('TextFont', window, 'Arial');
    Screen('TextSize', window, 80);
    Screen('TextColor', window, black);
    fprintf('Psychtoolbox window opened successfully.\n');
    ptbInitialized = true;
    % Define escape key
    KbName('UnifyKeyNames'); % Important for cross-platform compatibility
    escapeKey = KbName('ESCAPE');

catch ME
    ptbInitialized = false;
    warning('Could not initialize Psychtoolbox. Running without visual stimulus presentation. Error: %s\n', ME.message);
    fprintf('Please ensure Psychtoolbox is correctly installed and in your MATLAB path.\n');
    sca; % Close any open Psychtoolbox windows if error occurs
end
% --- 2. Experiment Start Message ---
fprintf('EEG Experiment Starting...\n');
fprintf('This experiment has %d blocks, with %d trials per block.\n', numBlocks, trialsPerBlock);
% Display a waiting screen with instructions
if ptbInitialized
    DrawFormattedText(window, 'Press any key to begin the experiment.\n(Press ESCAPE at any time to quit)', 'center', 'center', black);
    Screen('Flip', window);
end
fprintf('Press any key to begin the experiment.\n');
KbStrokeWait; % user input to start (Psychtoolbox function for keyboard input)
% Clear the screen after the key press
if ptbInitialized
    Screen('FillRect', window, grey); % Fill with background color
    Screen('Flip', window);
end
% Initialize a persistent timer for `toc_global`
initialGlobalTime = tic;
% --- 3. Main Experiment Loop ---
experimentAborted = false;
for block = 1:numBlocks
    if experimentAborted; break; end
    % instruction message at the beginning of every block
    if ptbInitialized
        if block == 1
            blockStartMessage = 'Welcome to the practice block!\n\nGet ready to begin.';
        elseif block == numBlocks
            blockStartMessage = sprintf('Starting the final block (%d of %d)!\n\nYou''re doing great, almost there!', block, numBlocks);
        else
            blockStartMessage = sprintf('Resuming Block %d of %d.\n\n Enjoy!', block, numBlocks);
        end

        blockStartText = sprintf('%s\n\n(Press ESCAPE to quit)', blockStartMessage);
        DrawFormattedText(window, blockStartText, 'center', 'center', black);
        Screen('Flip', window);

        % Check for escape during block start message display
        [~, ~, keyCode] = KbCheck;
        if keyCode(escapeKey)
            experimentAborted = true;
            break;
        end
        WaitSecs(2); % Show the message for 2 seconds
        Screen('FillRect', window, grey); % Clear the screen
        Screen('Flip', window);
    end
    if experimentAborted; break; end % Exit block loop if aborted
    fprintf('\n--- Starting Block %d of %d ---\n', block, numBlocks);

   % Send a marker for the start of the block via LSL
    markerValue = 99; % Block Start
    outlet.push_sample(markerValue); % LSL pushes marker with its own timestamp
    fprintf('LSL Marker %d sent (Block Start).\n', markerValue);

    % Randomize stimulus order for the current block and ensure 3
    % repetitions per stimulus
    numRepetitions = 2;
    blockStimulusIndices = repmat(1:numStimuli, 1, numRepetitions);
    % randomize
    blockStimulusIndices = blockStimulusIndices(randperm(length(blockStimulusIndices)));

    % Numeber of trials for main and practice blocks
    currentBlockTrials = trialsPerBlock;
    if block == 1
        currentBlockTrials = practiceTrialsPerPhase * 3; % 15 trials for each of 3 practice phases
    end
    % Generate randomized stimulus indices for the current block.
    % Ensure enough unique stimuli are available for randomization in practice.
    % For practice, we want 15 random trials from the full set of stimuli for each phase.
    if block == 1
        % Generate randomized indices for each practice phase
        practicePhase1Stimuli = randperm(numStimuli, practiceTrialsPerPhase);
        practicePhase2Stimuli = randperm(numStimuli, practiceTrialsPerPhase);
        practicePhase3Stimuli = randperm(numStimuli, practiceTrialsPerPhase);
        blockStimulusIndices = [practicePhase1Stimuli, practicePhase2Stimuli, practicePhase3Stimuli];
    else
        % For main blocks, stick to original repetition logic
        blockStimulusIndices = repmat(1:numStimuli, 1, numRepetitions);
        blockStimulusIndices = blockStimulusIndices(randperm(length(blockStimulusIndices)));
    end


    for trial = 1:currentBlockTrials % Use currentBlockTrials here
        if experimentAborted; break; end
        fprintf('  Block %d, Trial %d: Presenting stimulus...\n', block, trial);

        % Get the stimulus index for the current trial from the randomized list
        stimulusType = blockStimulusIndices(trial); % This is the stimulus original index
        currentSentence = allStimulitext{stimulusType};
        % --- Stimulus Presentation Logic based on Block and Trial ---
        markerValue = 0; % Initialize marker value
        markerLabel = ''; % Initialize marker label
        playAudio = false; % Flag for playing sentence audio
        showText = false; % Flag for showing sentence text
        showClickAndFixation = false; % Flag for showing click and fixation cross
        if block == 1 % Practice Block
            % MODIFICATION: Use practiceTrialsPerPhase for phase limits
            if trial <= practiceTrialsPerPhase
                % Practice Phase 1: Audio + Text
                if trial == 1 % Show instruction
                    instructionText = 'Practice Block 1/3: Audio and Text Simultaneously.\n\nPress any key to continue.\n(Press ESCAPE to quit)';
                    if ptbInitialized
                        DrawFormattedText(window, instructionText, 'center', 'center', black);
                        Screen('Flip', window);
                        KbStrokeWait;
                        Screen('FillRect', window, grey); Screen('Flip', window);
                    end
                    fprintf('%s\n', instructionText);
                end
                playAudio = true;
                showText = true;
                markerLabel = sprintf('Practice_AudioText_Sentence_%02d', stimulusType);
                markerValue = 200 + stimulusType; % 200 series for audio+text
            elseif trial <= 2 * practiceTrialsPerPhase
                % Practice Phase 2: Text Only
                if trial == practiceTrialsPerPhase + 1 % Show instruction before first trial of this phase
                    instructionText = 'Practice Block 2/3: Text Only (Inner Voice).\n\nPress any key to continue.\n(Press ESCAPE to quit)';
                    if ptbInitialized
                        DrawFormattedText(window, instructionText, 'center', 'center', black);
                        Screen('Flip', window);
                        KbStrokeWait;
                        Screen('FillRect', window, grey); Screen('Flip', window);
                    end
                    fprintf('%s\n', instructionText);
                end
                playAudio = false;
                showText = true;
                markerLabel = sprintf('Practice_VisualOnly_Sentence_%02d', stimulusType);
                markerValue = 100 + stimulusType; % 100 series for visual only
            else % last trials
                % Practice Phase 3: Text Only (with click and fixation for MI)
                if trial == (2 * practiceTrialsPerPhase) + 1 % Show instruction before first trial of this phase
                    instructionText = 'Practice Block 3/3: Text Only (Attempted Speech).\n\nPress any key to continue.\n(Press ESCAPE to quit)';
                    if ptbInitialized
                        DrawFormattedText(window, instructionText, 'center', 'center', black);
                        Screen('Flip', window);
                        KbStrokeWait;
                        Screen('FillRect', window, grey); Screen('Flip', window);
                    end
                    fprintf('%s\n', instructionText);
                end
                playAudio = false;
                showText = true;
                showClickAndFixation = true; % click and fixation for text-only practice
                markerLabel = sprintf('Practice_VisualOnly_Sentence_%02d', stimulusType);
                markerValue = 100 + stimulusType; % 100 series for text only
            end
        elseif block == 2
            % Block 2: Audio and Text Simultaneously
            playAudio = true;
            showText = true;
            markerLabel = sprintf('AudioText_Sentence_%02d', stimulusType);
            markerValue = 200 + stimulusType; % Marker for Audio Stimulus (e.g., 201-220)
        elseif block == 3 % Block 3: Visual stimulus (sentence only), SILENT READING, no fixation
            playAudio = false;
            showText = true;
            markerLabel = sprintf('VisualOnly_Sentence_%02d', stimulusType);
            markerValue = 100 + stimulusType; % Marker for Visual Stimulus (e.g., 101-120)
        else
            % Block 4: Visual stimulus (sentence only), MI with fixation
            playAudio = false;
            showText = true;
            showClickAndFixation = true; % Add click and fixation for MI
            markerLabel = sprintf('VisualOnly_Sentence_%02d', stimulusType);
            markerValue = 100 + stimulusType; % Marker for Visual Stimulus (e.g., 101-120)
        end
        % --- Execute main stimulus presentation (Text and/or Audio) ---
        if showText && ptbInitialized
            DrawFormattedText(window, currentSentence, 'center', 'center', black);
            Screen('Flip', window); % Show text
        end
        disp(['    ' currentSentence]); % Still display in command window for debugging/logging
        if playAudio
            try
                playAudioStimulus(stimulusType);
            catch ME
                warning('Could not play audio for stimulus %d. Error: %s\n', stimulusType, ME.message);
                % If audio fails, adjust marker to visual only if it was planned as audio+text
                if markerValue >= 200 && markerValue <= 220 % If it was an audio marker
                    markerValue = markerValue - 100; % Convert to visual only marker (e.g., 201 -> 101)
                    markerLabel = strrep(markerLabel, 'Audio', 'Visual'); % Update label
                    markerLabel = strrep(markerLabel, 'AudioText', 'VisualOnly'); % Update label for practice
                    warning('Adjusted marker to %d (%s) due to audio playback failure.', markerValue, markerLabel);
                end
            end
        end


        % Send a marker for stimulus onset via LSL
        outlet.push_sample(markerValue); % LSL pushes marker with its own timestamp
        fprintf('LSL Marker %d (%s) sent (Stimulus Onset).\n', markerValue, markerLabel);


        % Record stimulus onset time (using global timer for logging)
        stimulusOnsetTimestamp = toc_global(initialGlobalTime);

        % Section for button press detection and LSL marker
        % --- Wait for stimulus duration AND check for button press ---
        buttonPressedTimestamp = NaN; % Initialize to NaN (no press)

        startTime = GetSecs;
        while (GetSecs - startTime) < stimulusDuration
            [keyIsDown, ~, keyCode] = KbCheck;
            if keyIsDown
                if keyCode(escapeKey)
                    experimentAborted = true;
                    fprintf('    Stimulus duration interrupted by ESCAPE.\n');
                    break; % Exit while
                else % Any key other than escape is a valid response
                    if isnan(buttonPressedTimestamp) % Send marker only on the *first* press
                        buttonPressMarker = 1; % any button press
                        outlet.push_sample(buttonPressMarker);
                        fprintf('    LSL Marker %d (Button Press) sent.\n', buttonPressMarker);

                        buttonPressedTimestamp = toc_global(initialGlobalTime);
                        fprintf('    Button pressed at %f sec (relative).\n', buttonPressedTimestamp);
                    end
                end
            end
            % Give CPU a break to avoid maxing out a core
            WaitSecs(0.001);
        end

        % Clear the screen after main stimulus duration, before next visual element
        if ptbInitialized
            Screen('FillRect', window, grey); % Fill with background color
            Screen('Flip', window);
        end
        % --- Post-stimulus events for "Text Only" trials/blocks (Click + Fixation Cross) ---
        if showClickAndFixation
            % Send LSL marker for click
            clickMarkerValue = 50; % Soft click
            outlet.push_sample(clickMarkerValue);
            fprintf('LSL Marker %d (Soft Click) sent.\n', clickMarkerValue);

            % Play soft click
            playClickSound();
            % Draw fixation cross
            if ptbInitialized
                fixCrossLength = 20; %in pixels
                fixCrossLineWidth = 3;
                Screen('DrawLine', window, black, xCenter - fixCrossLength, yCenter, xCenter + fixCrossLength, yCenter, fixCrossLineWidth);
                Screen('DrawLine', window, black, xCenter, yCenter - fixCrossLength, xCenter, yCenter + fixCrossLength, fixCrossLineWidth);
                Screen('Flip', window);
            end
            % Wait for fixation cross duration with escape check
            if CheckForEscapeDuringWait(fixationCrossDuration, escapeKey, @() fprintf('Fixation cross interrupted by ESCAPE.\n'))
                experimentAborted = true;
                break; % Exit current trial loop
            end
            % Clear fixation cross
            if ptbInitialized
                Screen('FillRect', window, grey);
                Screen('Flip', window);
            end
        end
        % --- Log Data ---
        % Only log if the experiment was not aborted during this trial's stimulus presentation
        if ~experimentAborted
            experimentLog{logIndex, 1} = block;
            experimentLog{logIndex, 2} = trial;
            experimentLog{logIndex, 3} = stimulusType; % Log the 1-20 index
            experimentLog{logIndex, 4} = stimulusOnsetTimestamp;
            experimentLog{logIndex, 5} = currentSentence; % Log the actual sentence
            % **Modification: Logging button press timestamp**
            experimentLog{logIndex, 6} = buttonPressedTimestamp;
            logIndex = logIndex + 1;
        else
            fprintf('Trial %d aborted, not logging.\n', trial);
        end

        % --- Inter-Trial Interval (ITI) ---
        fprintf('    Inter-trial interval...\n');
        if CheckForEscapeDuringWait(itiDuration, escapeKey, @() fprintf('ITI interrupted by ESCAPE.\n'))
            experimentAborted = true;
            break;
        end
    end

    % BREAKS AT THE 74TH TRIAL OF MAIN BLOCKS (2,3,4)
    if trial == 74 && block > 1 && ~experimentAborted
        fprintf('\n--- Intermediate Break. Take a break now. ---\n');
        fprintf('When you''re ready, press any key to continue with Block %d.\n(Press ESCAPE to quit)\n', block);
        
        if ptbInitialized
            breakText = sprintf('Intermediate Break for Block %d.\n\nPress any key to continue.\n(Press ESCAPE to quit)', block);
            DrawFormattedText(window, breakText, 'center', 'center', black);
            Screen('Flip', window);
        end
        
        % Send a marker for the break via LSL
        markerValue = 97; % Marker for Intermediate Break
        outlet.push_sample(markerValue);
        fprintf('LSL Marker %d sent (Intermediate Break).\n', markerValue);
        
        % Wait for a key press to continue
        KbStrokeWait;
        
        % Clear the screen before resuming
        if ptbInitialized
            Screen('FillRect', window, grey);
            Screen('Flip', window);
        end
        fprintf('Resuming experiment...\n');
    end

    % --- Break between Blocks (key press to continue) ---
    if block < numBlocks && ~experimentAborted
        fprintf('\n--- End of Block %d. You can take a break now. ---\n', block);
        fprintf('When you are ready for Block %d, press any key to continue.\n(Press ESCAPE to quit)\n', block + 1);

        % Display break message on screen
        if ptbInitialized
            breakText = sprintf('End of Block %d.\nTake a break now.\n\nWhen ready for Block %d, press any key to continue.\n(Press ESCAPE to quit)', block, block + 1);
            DrawFormattedText(window, breakText, 'center', 'center', black);
            Screen('Flip', window);
        end
        % Send a marker for the end of the block/start of break via LSL
        markerValue = 98; % Marker for Block End/Break Start
        outlet.push_sample(markerValue);
        fprintf('LSL Marker %d sent (Block End/Break Start).\n', markerValue);

        % Wait for a key press to continue, replacing fixed breakDuration
        KbStrokeWait;

        % Check if ESCAPE was pressed while waiting for next block (manual check after KbStrokeWait)
        [~, ~, keyCode] = KbCheck;
        if keyCode(escapeKey)
            experimentAborted = true;
            fprintf('Experiment aborted by ESCAPE during break.\n');
            break; % Exit block loop
        end
        % Clear the screen before resuming
        if ptbInitialized
            Screen('FillRect', window, grey);
            Screen('Flip', window);
        end
        fprintf('Resuming experiment...\n');
    end
end
% --- 4. Experiment End ---
if experimentAborted
    fprintf('\n--- Experiment Aborted by User! ---\n');
    % Display message on screen if PTB initialized
    if ptbInitialized
        DrawFormattedText(window, 'Experiment Aborted!\nPress any key to close.', 'center', 'center', black);
        Screen('Flip', window);
        KbStrokeWait;
    end
else
    fprintf('\n--- Experiment Completed! ---\n');
    % Display experiment completion message
    if ptbInitialized
        DrawFormattedText(window, 'Experiment Completed!\nThank you for participating!', 'center', 'center', black);
        Screen('Flip', window);
        WaitSecs(3); % Show for 3 seconds
    end
end
% Send a marker for the end of the experiment via LSL
markerValue = 255; % Marker for Experiment End
outlet.push_sample(markerValue);
fprintf('LSL Marker %d sent (Experiment End).\n', markerValue);

% --- Psychtoolbox Cleanup ---
if ptbInitialized
    sca; % Close all screens and Psychtoolbox windows
    fprintf('Psychtoolbox windows closed.\n');
end
% --- Save Experiment Log ---
timestampStr = datestr(now, 'yyyymmdd_HHMMSS');
logFilename = sprintf('EEG_Experiment_Log_%s.mat', timestampStr);
% Save relevant experiment parameters and the log
save(logFilename, 'experimentLog', 'numBlocks', 'trialsPerBlock', 'stimulusDuration', 'itiDuration', 'numStimuli', 'allStimulitext', 'experimentAborted');
fprintf('Experiment log saved to %s\n', logFilename);
if ~experimentAborted
    fprintf('Thank you for participating!\n');
else
    fprintf('Experiment finished early due to abort.\n');
end
% --- Helper function to get a high-precision timestamp relative to experiment start ---
% This function takes the initial `tic` value to provide consistent relative timing.
function t_sec = toc_global(startTimeHandle)
    t_sec = toc(startTimeHandle);
end
% --- Helper function to play an audio stimulus based on its index ---
function playAudioStimulus(stimulusIndex)
    audioFilename = sprintf('stimuli/audio%d.wav', stimulusIndex);
    if exist(audioFilename, 'file') == 2
        [y, Fs] = audioread(audioFilename);
        % **Modification: Ensure audio is mono**
        numChannels = size(y, 2);
        if numChannels > 1
            y = mean(y, 2);
            warning('Converted audio stimulus from %d channels to mono (1 channel) for playback.', numChannels);
        end
        sound(y, Fs);
    else
        error('Audio file not found: %s. Please ensure all audioN.wav files are in the script directory.', audioFilename);
    end
end

% --- Function to play a soft click sound ---
function playClickSound()
        Fs_click = 44100;
        duration_click = 0.05;
        t_click = 0:1/Fs_click:duration_click;
        y_click = sin(2 * pi * 1000 * t_click) .* exp(-50 * t_click); % Damped sine wave
        y_click = y_click / max(abs(y_click)) * 0.5; % Normalize amplitude
        sound(y_click, Fs_click);
end
% --- Function to wait while checking for an escape key press ---
function aborted = CheckForEscapeDuringWait(duration, escapeKeyCode, printMsgFunc)
    startTime = GetSecs;
    aborted = false;
    while (GetSecs - startTime) < duration
        [keyIsDown, ~, keyCode] = KbCheck;
        if keyIsDown && keyCode(escapeKeyCode)
            aborted = true;
            if nargin > 2 && ~isempty(printMsgFunc)
                printMsgFunc();
            end
            break;
        end
        % Give CPU a break to avoid maxing out a core
        WaitSecs(0.001);
    end
end