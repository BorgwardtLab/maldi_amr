#!/usr/bin/env bash
#
# The purpose of this script is to submit feature importance calculation
# jobs to an LSF system in order to speed up job processing.

# Main command to execute for all combinations created in the script
# below. The space at the end of the string is important.
MAIN="poetry run python ../single_prediction_scores.py "

# Try to be smart: if `bsub` does *not* exist on the system, we just
# pretend that it is an empty command.
if [ -x "$(command -v bsub)" ]; then
  BSUB='bsub -W 3:59 -o "feature_importance_values_%J.out" -R "rusage[mem=32000]"'
fi

# Evaluates its first argument either by submitting a job, or by
# executing the command without parallel processing.
run() {
  if [ -z "$BSUB" ]; then
    eval "$1";
  else
    eval "${BSUB} $1";
  fi
}

# S. aureus jobs
for code in "68025e38-6ed8-4250-8fdd-555c33a35c6d_MALDI1" \
            "c9aac38c-112a-4226-bc34-e3e39d07d046_MALDI1" \
            "7b54314c-da28-4310-a57f-1f211cc8b54e_MALDI1" \
            "68025e38-6ed8-4250-8fdd-555c33a35c6d_MALDI1" \
            "c9aac38c-112a-4226-bc34-e3e39d07d046_MALDI1" \
            "92c8e066-2535-45f7-80cf-3e5f158d6357_MALDI1" \
            "26870b68-5bf8-4c7e-afa2-a86a4e4ebdd5_MALDI2" \
            "65269c5a-b580-40fb-b5a2-cb17e012213d" \
            "e79e3ace-9d85-4182-a86b-6092685d3ad3_MALDI1" \
            "188906e2-c42d-4508-8ebc-0958a932a21a_MALDI2" \
            "1dec2cc8-8440-4bb2-8e12-cf2ad08c1946_MALDI1" \
            "d555a794-ae51-428f-a76c-97ec2c2fe5c3_MALDI1" \
            "483adf87-857c-4d23-a116-bacd9c961097_MALDI1" \
            "4087ecbd-c4bd-4264-97db-c0216f32ed1d_MALDI2" \
            "94945229-059b-47a9-ab4f-33560ac3383b" \
            "dcafe547-001b-4574-92ee-48eba6a16d3c_MALDI1" \
            "0c30e1ae-3bb4-4163-b0a7-0af5396c3d29_MALDI1" \
            "062cef28-46c5-45ed-abf3-327bc769503f_MALDI1" \
            "eaf3e1dc-7785-4ebc-b894-6f6c39f1eede_MALDI1" \
            "55485d32-9055-4e75-af7b-4f72f4e367f5_MALDI1" \
            "e4d7e01f-c39d-4c79-8700-0323abdd7908_MALDI1" \
            "66c1b675-9a28-4a0d-bcf4-b157219ce152_MALDI2" \
            "8ebdecc1-1aa0-444f-9118-4c47b0762891" \
            "6f020bc2-4ca3-4d3a-9072-5a9a31a2aef4_MALDI2" \
            "7e5d8cda-65e4-4dd0-ae26-f5a39eee17fb_MALDI2" \
            "128ca36e-ac41-44d6-82a1-49bd0655cb7f" \
            "143fd5ed-4f2b-43c8-930a-16a659e4052c" \
            "4a55e482-da27-41ba-a667-73d0a6652924_MALDI1" \
            "7e69664e-b2a5-453e-959f-f031fecdc2f0_MALDI2" \
            "d65a657e-2418-4eba-948d-9a52ab4b3c91_MALDI1" \
            "32367ade-6970-42f9-a518-132099279dbe_MALDI1" \
            "5c4b911b-44b1-42ee-a021-6cc658d69b94_MALDI1" \
            "0d6eba5b-591d-4309-8966-dc1ff68a1ee3_MALDI1" \
            "63833212-cbbe-4795-b9f3-d9b0d6363f9f_MALDI1" \
            "6b3f5639-a39b-49df-ad00-0c3c52a4f8e7_MALDI1" \
            "f3fb9c8a-e05c-4b70-9cdf-7ff06f16dcd8_MALDI2" \
            "c5f84e04-b950-4a4b-b612-b8715711d40e_MALDI2" \
            "476d5980-d3c3-401c-a7be-50518a470e1f_MALDI1" \; do
    for FILE in $(find ../../results/fig4_curves_per_species_and_antibiotics/lr/ -name "*aureus*Oxacillin*164.json"); do
        run "${MAIN} --input $code ${FILE}"
    done
done

## K. pneumoniae jobs
#for FILE in $(find ../../results/fig4_curves_per_species_and_antibiotics/lr/ -name "*pneu*Meropenem*"); do
#  run "${MAIN} ${FILE}"
#done
#
## E. coli jobs
#for FILE in $(find ../../results/fig4_curves_per_species_and_antibiotics/lr/ -name "*coli*Ceftriaxone*"); do
#  run "${MAIN} ${FILE}"
#done
