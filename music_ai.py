import glob
import numpy as np
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# 1. Dataset se MIDI files read karna
midi_files = glob.glob("dataset/*.mid")
notes = []

print("Files read ho rahi hain...")
for file in midi_files:
    try:
        midi = converter.parse(file)
        # Instrument tracks separate karna
        parts = instrument.partitionByInstrument(midi)
        elements = parts.parts[0].recurse() if parts else midi.flat.notes

        for el in elements:
            if isinstance(el, note.Note):
                notes.append(str(el.pitch))
            elif isinstance(el, chord.Chord):
                notes.append('.'.join(str(n) for n in el.normalOrder))
    except Exception:
        continue

# 2. Vocabulary aur Inputs tayar karna
vocab = sorted(list(set(notes)))
note_to_int = {n: i for i, n in enumerate(vocab)}

seq_len = 100
X = []
y = []

for i in range(len(notes) - seq_len):
    in_seq = notes[i : i + seq_len]
    out_note = notes[i + seq_len]
    X.append([note_to_int[n] for n in in_seq])
    y.append(note_to_int[out_note])

num_samples = len(X)
X_norm = np.reshape(X, (num_samples, seq_len, 1)) / float(len(vocab))

# One-hot encoding manual tareeke se
y_encoded = np.zeros((num_samples, len(vocab)))
for i, val in enumerate(y):
    y_encoded[i, val] = 1.0

# 3. Simple LSTM Model
model = Sequential()
model.add(LSTM(256, input_shape=(seq_len, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dense(len(vocab), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
print("Model training start ho rahi hai...")
model.fit(X_norm, y_encoded, epochs=10, batch_size=64)

# 4. Naya Music Generate Karna
start_idx = np.random.randint(0, len(X) - 1)
pattern = X[start_idx]
predicted_notes = []
int_to_note = {i: n for i, n in enumerate(vocab)}

for _ in range(200):
    inp = np.reshape(pattern, (1, len(pattern), 1)) / float(len(vocab))
    pred = model.predict(inp, verbose=0)
    
    idx = np.argmax(pred)
    predicted_notes.append(int_to_note[idx])
    
    pattern.append(idx)
    pattern = pattern[1:]

# 5. Output ko MIDI file mein save karna
offset = 0
output_stream = []

for p in predicted_notes:
    if ('.' in p) or p.isdigit():
        chord_notes = p.split('.')
        notes_list = [note.Note(int(n)) for n in chord_notes]
        c = chord.Chord(notes_list)
        c.offset = offset
        output_stream.append(c)
    else:
        n = note.Note(p)
        n.offset = offset
        output_stream.append(n)
    offset += 0.5

midi_out = stream.Stream(output_stream)
midi_out.write('midi', fp='output.mid')
print("Complete! Music file 'output.mid' ke naam se save ho gayi hai.")
