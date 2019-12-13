# LSTM
import tensorflow as tf
import numpy as np
import mnist as data
import sys, os, time, imageio

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

TF_TYPE=tf.float32 

RNN_DEPTH = 1
RNN_WIDTH = 5

LSTM_HDIM = 128

GLIMPSE_SIZE = 8 # single glimpse edge
GLIMPSE_HDIM = 3 * GLIMPSE_SIZE * GLIMPSE_SIZE

GLIMPSE_SCALE_1 = 1
GLIMPSE_SCALE_2 = 2
GLIMPSE_SCALE_3 = 3

ACTION_DIM = 10 # 10 classes

LOCATION_XDIM = 2 # [x sampled, y sampled]
LOCATION_HDIM = 4 # [x mean, y mean, x sd, y sd]

MAX_STEP_MEAN = 1.
MAX_STEP_SD = 1.

EPSILON = 1e-10

SAVED_FOLDER = "saved"
SAVED_MODEL = SAVED_FOLDER + "/ram.ckpt"
SAVED_STEPS = 1000

TRAIN_RATE = 0.001
TRAIN_MOMENTUM = 0.5 * 0

def random_matrix(xdim, ydim):
  return np.random.uniform(low=-0.2, high=0.2, size=(xdim, ydim))

def ones_matrix(xdim, ydim):
  return np.ones((xdim, ydim))

def zeros_matrix(xdim, ydim):
  return np.zeros((xdim, ydim))

def print_matrix(m):
  for i in xrange(1):  
    sys.stdout.write(" r\c ")
    for y in xrange(m.shape[1]):
      sys.stdout.write("%3d " % y)
    print ""
    for x in xrange(m.shape[0]):
      sys.stdout.write("%3d  " % x)
      for y in xrange(m.shape[1]):
        sys.stdout.write("%3d " % m[x][y])
      print ""

def movie_frame(plot):
  fname = "%s/plot-frame.png" % SAVED_FOLDER
  plot.savefig(fname)
  frame = imageio.imread(fname)
  os.remove(fname)
  return frame

def movie_save(frames):
  fname = "%s/plot-movie.gif" % SAVED_FOLDER
  imageio.mimsave(fname, frames, duration=0.1)

# Glimpse sensor
def glimpse_average(img, x1, x2, y1, y2):
  scale = (x2 - x1) * (y2 - y1)
  total = 0
  for x in xrange(x1, x2):
    for y in xrange(y1, y2):
      if x >= 0 and x < img.shape[0] and y >= 0 and y < img.shape[1]:
        total += img[x][y]

  return round(float(total) / scale)

def glimpse_rect(lx, ly, block_dim):
  lx = int(round(lx))
  ly = int(round(ly))
  
  x = lx - GLIMPSE_SIZE * block_dim / 2
  y = ly - GLIMPSE_SIZE * block_dim / 2
  dx = GLIMPSE_SIZE * block_dim
  dy = GLIMPSE_SIZE * block_dim
  return x, y, dx, dy

def glimpse_patch(img, lx, ly, block_dim):  
  offset_x = lx - GLIMPSE_SIZE * block_dim / 2
  offset_y = ly - GLIMPSE_SIZE * block_dim / 2
  patch = zeros_matrix(GLIMPSE_SIZE, GLIMPSE_SIZE)
  for x in xrange(GLIMPSE_SIZE):
    for y in xrange(GLIMPSE_SIZE):
      x1 = offset_x + x * block_dim
      x2 = offset_x + (x + 1) * block_dim
      y1 = offset_y + y * block_dim
      y2 = offset_y + (y + 1) * block_dim
      patch[x][y] = glimpse_average(img, x1, x2, y1, y2)
  return patch / 255.0

def glimpse(img, ix, iy):
  ix = int(round(ix))
  iy = int(round(iy))

  g1 = glimpse_patch(img, ix, iy, GLIMPSE_SCALE_1)
  g2 = glimpse_patch(img, ix, iy, GLIMPSE_SCALE_2)
  g3 = glimpse_patch(img, ix, iy, GLIMPSE_SCALE_3)
  
  return np.reshape(
    np.stack((g1, g2, g3)), 
    (3*GLIMPSE_SIZE*GLIMPSE_SIZE, 1))

def print_glimpse(glimpse):
  glimpses = np.split(glimpse, 3)
  for g in xrange(len(glimpses)):
    print("Gmlimpse: %d" % g)
    print_matrix(glimpses[g].reshape((GLIMPSE_SIZE, GLIMPSE_SIZE)) * 255)
        
# LSTM with peepholes
class LSTM(object):
  def __init__(self, x, prev):
    xdim = x.shape[0]
    hdim = LSTM_HDIM

    if isinstance(prev, LSTM):

      ht_1 = prev.h
      ct_1 = prev.c

      self.Wix = prev.Wix
      self.Wih = prev.Wih
      self.Wic = prev.Wic
      self.bi = prev.bi

      self.Wfx = prev.Wfc
      self.Wfh = prev.Wfh
      self.Wfc = prev.Wfc
      self.bf = prev.bf

      self.Wcx = prev.Wcx
      self.Wch = prev.Wch
      self.bc = prev.bc

      self.Wox = prev.Wox
      self.Woh = prev.Woh
      self.Woc = prev.Woc
      self.bo = prev.bo

    else:

      ht_1 = tf.constant(zeros_matrix(hdim, 1), dtype=TF_TYPE)
      ct_1 = tf.constant(zeros_matrix(hdim, 1), dtype=TF_TYPE)

      self.Wix = tf.Variable(random_matrix(hdim, xdim), dtype=TF_TYPE) # h by x
      self.Wih = tf.Variable(random_matrix(hdim, hdim), dtype=TF_TYPE) # h by h
      self.Wic = tf.Variable(random_matrix(hdim, 1), dtype=TF_TYPE) # h
      self.bi = tf.Variable(random_matrix(hdim, 1), dtype=TF_TYPE) # h

      self.Wfx = tf.Variable(random_matrix(hdim, xdim), dtype=TF_TYPE) # h by x
      self.Wfh = tf.Variable(random_matrix(hdim, hdim), dtype=TF_TYPE) # h by h
      self.Wfc = tf.Variable(random_matrix(hdim, 1), dtype=TF_TYPE) # h
      self.bf = tf.Variable(random_matrix(hdim, 1), dtype=TF_TYPE) # h    

      self.Wcx = tf.Variable(random_matrix(hdim, xdim), dtype=TF_TYPE) # h by x
      self.Wch = tf.Variable(random_matrix(hdim, hdim), dtype=TF_TYPE) # h by h
      self.bc = tf.Variable(random_matrix(hdim, 1), dtype=TF_TYPE) # h    

      self.Wox = tf.Variable(random_matrix(hdim, xdim), dtype=TF_TYPE) # h by x
      self.Woh = tf.Variable(random_matrix(hdim, hdim), dtype=TF_TYPE) # h by h
      self.Woc = tf.Variable(random_matrix(hdim, 1), dtype=TF_TYPE) # h
      self.bo = tf.Variable(random_matrix(hdim, 1), dtype=TF_TYPE) # h   

    self.i = tf.sigmoid(
      tf.matmul(self.Wix, x) + 
      tf.matmul(self.Wih, ht_1) + 
      tf.multiply(self.Wic, ct_1) + 
      self.bi)

    self.f = tf.sigmoid(
      tf.matmul(self.Wfx, x) + 
      tf.matmul(self.Wfh, ht_1) + 
      tf.multiply(self.Wfc, ct_1) +
      self.bf)
    
    self.c = tf.multiply(self.f, ct_1) + \
      tf.multiply(self.i, tf.tanh(
        tf.matmul(self.Wcx, x) + 
        tf.matmul(self.Wch, ht_1) + 
        self.bc))

    self.o = tf.sigmoid(
      tf.matmul(self.Wox, x) + 
      tf.matmul(self.Woh, ht_1) + 
      tf.multiply(self.Woc, self.c) + 
      self.bo)

    self.h = tf.multiply(self.o, tf.tanh(self.c)) # h

  def variables(self):
    v = list()

    v.append(self.Wix)
    v.append(self.Wih)
    v.append(self.Wic)
    v.append(self.bi)

    v.append(self.Wfx)
    v.append(self.Wfh)
    v.append(self.Wfc)
    v.append(self.bf)

    v.append(self.Wcx)
    v.append(self.Wch)
    v.append(self.bc)

    v.append(self.Wox)
    v.append(self.Woh)
    v.append(self.Woc)
    v.append(self.bo)    

    return v
        
# Deep Projected LSTM
class DPLSTM(object):
  def __init__(self, x, prev):
    hdim = LSTM_HDIM
    xdim = x.shape[0]

    if isinstance(prev, DPLSTM):
      t = 1 # indicate time > 0
      self.Wh = prev.Wh
      self.Wx = prev.Wx
      self.b = prev.b
    else:
      t = 0
      self.Wh = tf.Variable(random_matrix(hdim, hdim), dtype=TF_TYPE) # h by h
      self.Wx = tf.Variable(random_matrix(hdim, xdim), dtype=TF_TYPE) # h by x
      self.b = tf.Variable(random_matrix(hdim, 1), dtype=TF_TYPE) # h    

    self.lstm = list()
    for d in xrange(RNN_DEPTH):
      x_i = x if d == 0 else self.lstm[d-1].h
      h_i = None if t == 0 else prev.lstm[d].h
      self.lstm.append(LSTM(x_i, h_i))

    self.h = tf.nn.relu(
      tf.matmul(self.Wh, self.lstm[RNN_DEPTH-1].h) + 
      tf.matmul(self.Wx, x) + 
      self.b)

  def variables(self):
    v = list()

    v.append(self.Wh)
    v.append(self.Wx)
    v.append(self.b)

    for d in xrange(RNN_DEPTH):
      v += self.lstm[d].variables()

    return v

# Glimpse network
class Glimpse(object):
  def __init__(self, l, g, prev):

    ldim = l.shape[0]
    gdim = g.shape[0]
    hdim = GLIMPSE_HDIM

    if isinstance(prev, Glimpse):
      self.Wl1 = prev.Wl1
      self.Wg1 = prev.Wg1
      self.bl1 = prev.bl1
      self.bg1 = prev.bg1
      self.Wl2 = prev.Wl2
      self.Wg2 = prev.Wg2
      self.b2 = prev.b2
    else:
      self.Wl1 = tf.Variable(random_matrix(hdim, ldim), dtype=TF_TYPE) # h by l
      self.Wg1 = tf.Variable(random_matrix(hdim, gdim), dtype=TF_TYPE) # h by g
      self.bl1 = tf.Variable(random_matrix(hdim, 1), dtype=TF_TYPE) # h
      self.bg1 = tf.Variable(random_matrix(hdim, 1), dtype=TF_TYPE) # h
      self.Wl2 = tf.Variable(random_matrix(hdim, hdim), dtype=TF_TYPE) # h by h
      self.Wg2 = tf.Variable(random_matrix(hdim, hdim), dtype=TF_TYPE) # h by h
      self.b2 = tf.Variable(random_matrix(hdim, 1), dtype=TF_TYPE) # h

    self.l = tf.nn.relu(tf.matmul(self.Wl1, l) + self.bl1)
    self.g = tf.nn.relu(tf.matmul(self.Wg1, g) + self.bg1)
    self.h = tf.nn.relu(
      tf.matmul(self.Wl2, self.l) + tf.matmul(self.Wg2, self.g) + self.b2)

  def variables(self):
    v = list()

    v.append(self.Wl1)
    v.append(self.Wg1)
    v.append(self.bl1)
    v.append(self.bg1)
    v.append(self.Wl2)
    v.append(self.Wg2)
    v.append(self.b2)

    return v
       
# Recurrent Attention Model
class RAM(object):
  def __init__(self, l, g, prev):
    hdim = LSTM_HDIM
    ldim = LOCATION_HDIM / 2
    adim = ACTION_DIM

    if isinstance(prev, RAM):
      self.Wa = prev.Wa
      self.Wu = prev.Wu
      self.Ws = prev.Ws
      self.Wv = prev.Wv
      self.ba = prev.ba
      self.bu = prev.bu
      self.bs = prev.bs
      self.bv = prev.bv
    else:
      self.Wa = tf.Variable(random_matrix(adim, hdim), dtype=TF_TYPE) # h by l
      self.Wu = tf.Variable(random_matrix(ldim, hdim), dtype=TF_TYPE) # h by l
      self.Ws = tf.Variable(random_matrix(ldim, hdim), dtype=TF_TYPE) # h by l
      self.Wv = tf.Variable(random_matrix(1, hdim), dtype=TF_TYPE) # 1 by h
      self.ba = tf.Variable(random_matrix(adim, 1), dtype=TF_TYPE) # h
      self.bu = tf.Variable(random_matrix(ldim, 1), dtype=TF_TYPE) # h
      self.bs = tf.Variable(random_matrix(ldim, 1), dtype=TF_TYPE) # h
      self.bv = tf.Variable(random_matrix(1, 1), dtype=TF_TYPE) # 1

    self.glimpse = Glimpse(l, g, prev.glimpse if prev is not None else None)
    self.core = DPLSTM(self.glimpse.h, prev.core if prev is not None else None)

    self.a = tf.nn.softmax(tf.matmul(self.Wa, self.core.h) + self.ba, dim=0)
    self.u = tf.tanh(tf.matmul(self.Wu, self.core.h) + self.bu)
    self.s = tf.sigmoid(tf.matmul(self.Ws, self.core.h) + self.bs)
    self.v = tf.nn.sigmoid(tf.matmul(self.Wv, self.core.h) + self.bv)

    lu = MAX_STEP_MEAN * self.u
    ls = MAX_STEP_SD * self.s + 0.5 # 0.5 to keep l_prob < 1

    normal = tf.distributions.Normal(loc=lu, scale=ls)

    # sample location and probability
    self.l_samp = normal.sample([1])
    self.l_prob = normal.prob(self.l_samp)

    self.l_samp = tf.reshape(self.l_samp, (2,1))
    self.l_prob = tf.reshape(self.l_prob, (2,1))
    
  def variables(self):
    v = list()

    v.append(self.Wa)
    v.append(self.Wu)
    v.append(self.Ws)
    v.append(self.Wv)
    v.append(self.ba)
    v.append(self.bu)
    v.append(self.bs)
    v.append(self.bv)

    v += self.glimpse.variables()
    v += self.core.variables()

    return v

print "Processing options..."
train_model = False
restore_model = False
demo_mode = False

for arg in sys.argv:
  if arg == "-r":
    restore_model = True
  if arg == "-t":
    train_model = True
  if arg == "-d":
    demo_mode = True

if not train_model:
  restore_model = True

print("MNIST test")
train_images = data.train_images()
train_labels = data.train_labels()
test_images = data.test_images()
test_labels = data.test_labels()

image = train_images[0]
print_matrix(image)
pos = np.array([image.shape[0]/2, image.shape[1]/2]).reshape((2,1))

print("Glimpse pos: %dx%d" % (pos[0], pos[1]))
print_glimpse(glimpse(image, pos[0], pos[1]))
      
if train_model:
  TITLE = "RAM training"
else:  
  TITLE = "RAM testing"
print TITLE

# timeline
a = list() # network
g = list() # glimpse
r = list() # reward
l = list() # location
one_hot_class = tf.placeholder(shape=(ACTION_DIM, 1), dtype=TF_TYPE) # class
one_hot_array = zeros_matrix(ACTION_DIM, 1)

for t in xrange(RNN_WIDTH):
  l.append(tf.placeholder(shape=(LOCATION_XDIM, 1), dtype=TF_TYPE))
  g.append(tf.placeholder(shape=(GLIMPSE_HDIM, 1), dtype=TF_TYPE))
  r.append(tf.placeholder(shape=(1, 1), dtype=TF_TYPE))
  ram = RAM(l[t], g[t], None if t == 0 else a[t-1])
  a.append(ram)
ram_vars = ram.variables()
ram_names = [var.op.name for var in ram_vars]

# value estimate through all times
v_est = tf.add_n([tf.square(act.v - r[-1]) for act in a])
J1 = tf.reduce_sum(v_est)

# action cost at the end
#a_err = [tf.log(A.a + EPSILON) * one_hot_class for A in a]
#J2 = - tf.reduce_sum(tf.add_n(a_err)) / RNN_WIDTH
J2 = - tf.reduce_sum(tf.log(a[-1].a + EPSILON) * one_hot_class)

# location cost through all times
B = tf.stop_gradient(tf.add_n([act.v for act in a])) / RNN_WIDTH
l_err = [tf.log(A.l_prob + EPSILON) * (R) for A,R in zip(a,r)]
J3 = - tf.reduce_sum(tf.add_n(l_err)) / RNN_WIDTH

# total cost
cost = J1 + J2 + J3
        
# partial training with SGD + momemntum
grad = tf.gradients(cost, a[-1].bu)
if TRAIN_MOMENTUM > 0:
  train = tf.train.MomentumOptimizer(TRAIN_RATE, TRAIN_MOMENTUM).minimize(cost)
else:
  train = tf.train.GradientDescentOptimizer(TRAIN_RATE).minimize(cost)
with tf.control_dependencies([train]):
  update = tf.constant(0)
  
# fetches and feeds
fetches = list();
feeds = list();
for t in xrange(RNN_WIDTH):
  fetches.append(a[t].a)
  fetches.append(a[t].l_samp)
  fetches.append(a[t].v)
  feeds.append(l[t])
  feeds.append(g[t])
  feeds.append(r[t])

if train_model:
  ttest = [J1, J2, J3]
  feeds.append(one_hot_class)
  fetches.append(grad)
  fetches.append(ttest)
  fetches.append(cost)
  fetches.append(update)

# init tf graph
init = tf.global_variables_initializer()
saver = tf.train.Saver(dict(zip(ram_names, ram_vars)))
sess = tf.Session()
sess.run(init)

# restore the model
if restore_model:
  saver.restore(sess, SAVED_MODEL)
  print "Model restored from:", SAVED_MODEL

# setup demo
if demo_mode:
  idata = zeros_matrix(image.shape[0], image.shape[1])
  gdata = zeros_matrix(GLIMPSE_SIZE, GLIMPSE_SIZE)
  patch = None
  movie = list()
   
  plt.ion()
  fig = plt.figure(TITLE)
  fig.suptitle("Activity...", fontsize=16)
  gsp = gridspec.GridSpec(2, 2)

  img_plot = fig.add_subplot(gsp[0])
  img_plot.set_title("Image label ...")
  img = img_plot.imshow(idata, cmap='gist_gray_r', vmin=0, vmax=255)

  gl1_plot = fig.add_subplot(gsp[1])
  gl2_plot = fig.add_subplot(gsp[2])
  gl3_plot = fig.add_subplot(gsp[3])

  gl1_plot.set_title("Zoom level 1")
  gl2_plot.set_title("Zoom level 2")
  gl3_plot.set_title("Zoom level 3")

  gl1 = gl1_plot.imshow(gdata, cmap='gist_gray_r', vmin=0, vmax=255)
  gl2 = gl2_plot.imshow(gdata, cmap='gist_gray_r', vmin=0, vmax=255)
  gl3 = gl3_plot.imshow(gdata, cmap='gist_gray_r', vmin=0, vmax=255)

  #gsp.tight_layout(fig)
  fig.tight_layout()
  fig.subplots_adjust(top=0.85)

# data
the_images = train_images if train_model else test_images
the_labels = train_labels if train_model else test_labels

IMAGE_COUNT = len(the_images)
TRAIN_STEPS = 1
print "Data Set:", IMAGE_COUNT

reward_all = 0.0
confidence_all = 0.0

# loop
for step in xrange(TRAIN_STEPS):
  checkpoint = 0
  reward_sum = 0.0
  confidence_sum = 0.0
  randomize_img = True if demo_mode else train_model and np.random.rand(1) > 0.5

  for i in xrange(IMAGE_COUNT):
    ii = np.random.randint(0, IMAGE_COUNT) if randomize_img else i
      
    image = the_images[ii]
    label = the_labels[ii]

    loc = zeros_matrix(2, 1)
    pos = np.array([image.shape[0]/2, image.shape[1]/2]).reshape((2,1))
    gli = glimpse(image, pos[0], pos[1])
    upd = dict()

    if demo_mode:
      fig.suptitle("Looking...", fontsize=16)
      img_plot.set_title("Image label %d" % label)
      img.set_data(image * -1 + 255)
    
    h = sess.partial_run_setup(fetches, feeds)
    for t in xrange(RNN_WIDTH):
      #print("Glimpse loc: %fx%f" % (loc[0], loc[1]))
      #print("Glimpse pos: %dx%d" % (pos[0], pos[1]))
      #print_glimpse(gli)
      if demo_mode:
        glimpses = [gl.reshape((GLIMPSE_SIZE, GLIMPSE_SIZE)) for gl in np.split(gli, 3)]
        gl1.set_data(glimpses[0] * -255 + 255)
        gl2.set_data(glimpses[1] * -255 + 255)
        gl3.set_data(glimpses[2] * -255 + 255)

        if patch is not None:
          for p in patch:
            p.remove()

        glr1 = glimpse_rect(pos[0], pos[1], GLIMPSE_SCALE_1)
        glr2 = glimpse_rect(pos[0], pos[1], GLIMPSE_SCALE_2)
        glr3 = glimpse_rect(pos[0], pos[1], GLIMPSE_SCALE_3)

        patch = [
          patches.Rectangle((glr1[1], glr1[0]), glr1[2], glr1[3], 
            facecolor="none", edgecolor="yellow"),
          patches.Rectangle((glr2[1], glr2[0]), glr2[2], glr2[3], 
            facecolor="none", edgecolor="yellow"),
          patches.Rectangle((glr3[1], glr3[0]), glr3[2], glr3[3], 
            facecolor="none", edgecolor="yellow"),
        ]
        for p in patch:
          img_plot.add_patch(p)

        fig.canvas.draw()
        movie.append(movie_frame(plt))
        time.sleep(0.1)

      # run time step
      loc, act, value = sess.partial_run(
        h, [a[t].l_samp, a[t].a, a[t].v], feed_dict={l[t]: loc, g[t]: gli})
      #print "a.a:"
      #print act
      
      # sample location and glimpse
      loc = loc.flatten()
      pos = pos.flatten()
      #print "sampled loc"
      #print loc
      pos = pos + loc
      #print "sampled pos"
      #print pos
      pos = np.reshape(pos, (2, 1))
      loc = np.reshape(loc, (2, 1))
      gli = glimpse(image, pos[0], pos[1])
      upd[r[t]] = np.array([0]).reshape((1,1))
    
    # sample action
    action = np.reshape(act, (1, ACTION_DIM))[0]
    if train_model:
      action = np.random.choice(ACTION_DIM, 1, p=action)[0]
    else:
      action = np.argmax(act)
    confid = act[action][0]
    
    # create one-hot label
    one_hot_array.fill(0)
    one_hot_array[label] = 1
        
    # set final reward and label
    reward = 1 if action == label else 0
    reward_sum += reward
    reward_all += reward
    confidence_sum += confid
    confidence_all += confid

    # demo
    if demo_mode:
      estimate = "%d with %.0f %% confidence" % (action, 100 * confid)
      fig.suptitle(estimate, fontsize=16)

      fig.canvas.draw()
      frame = movie_frame(plt)
      for _ in xrange(20):
        movie.append(frame)
      if i != 0 and i % 10 == 0:      
        movie_save(movie)
        del movie[:]
      time.sleep(2)
    
    # update network
    if train_model:
      reward = np.array([reward]).reshape((1,1))
      for rh in r:
        upd[rh] = reward
      upd[one_hot_class] = one_hot_array
      error, u, tt = sess.partial_run(h, [cost, update, ttest], feed_dict=upd)
      #print "ttest:", tt

    # report and save model
    if (i+1) % SAVED_STEPS == 0:
      checkpoint += 1
      if train_model:
        save_path = saver.save(sess, SAVED_MODEL)
      print "Step:", step + 1, "|",
      print "Checkpoint:", checkpoint, "|",
      print "Random:", randomize_img, "|",
      print "Accuracy: %.1f %% |" % (100 * reward_sum / SAVED_STEPS),
      print "Confidence: %.1f %%" % (100 * confidence_sum / SAVED_STEPS)
      reward_sum = 0.0
      confidence_sum = 0.0
      
print "Total Accuracy: %.1f %% |" % (100 * reward_all / TRAIN_STEPS/ IMAGE_COUNT),
print "Total Confidence: %.1f %%" % (100 * confidence_all / TRAIN_STEPS/ IMAGE_COUNT)

