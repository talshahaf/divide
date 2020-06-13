from PIL import Image
import pytesseract
import os, sys, re, pprint, csv, io, itertools, pickle
import numpy as np
import cv2

if os.name == 'nt':
    os.environ['PATH'] += r';E:\Tesseract-OCR'
    os.environ['TESSDATA_PREFIX'] = r'E:\Tesseract-OCR\tessdata'

# Somehow I found the value of `gamma=1.2` to be the best in my case
def adjust_gamma(image, gamma=1.2):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)
    
# These are probably the only important parameters in the
# whole pipeline (steps 0 through 3).
BLOCK_SIZE = 40
DELTA = 25

# Do the necessary noise cleaning and other stuffs.
# I just do a simple blurring here but you can optionally
# add more stuffs.
def preprocess(image):
    image = cv2.medianBlur(image, 3)
    return 255 - image

# Again, this step is fully optional and you can even keep
# the body empty. I just did some opening. The algorithm is
# pretty robust, so this stuff won't affect much.
def postprocess(image):
    kernel = np.ones((3,3), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image

# Just a helper function that generates box coordinates
def get_block_index(image_shape, yx, block_size): 
    y = np.arange(max(0, yx[0]-block_size), min(image_shape[0], yx[0]+block_size))
    x = np.arange(max(0, yx[1]-block_size), min(image_shape[1], yx[1]+block_size))
    return tuple(np.meshgrid(y, x))

# Here is where the trick begins. We perform binarization from the 
# median value locally (the img_in is actually a slice of the image). 
# Here, following assumptions are held:
#   1.  The majority of pixels in the slice is background
#   2.  The median value of the intensity histogram probably
#       belongs to the background. We allow a soft margin DELTA
#       to account for any irregularities.
#   3.  We need to keep everything other than the background.
#
# We also do simple morphological operations here. It was just
# something that I empirically found to be "useful", but I assume
# this is pretty robust across different datasets.
def adaptive_median_threshold(img_in):
    med = np.median(img_in)
    img_out = np.zeros_like(img_in)
    img_out[img_in - med < DELTA] = 255
    kernel = np.ones((3,3),np.uint8)
    img_out = 255 - cv2.dilate(255 - img_out,kernel,iterations = 2)
    return img_out

# This function just divides the image into local regions (blocks),
# and perform the `adaptive_mean_threshold(...)` function to each
# of the regions.
def block_image_process(image, block_size):
    out_image = np.zeros_like(image)
    for row in range(0, image.shape[0], block_size):
        for col in range(0, image.shape[1], block_size):
            idx = (row, col)
            block_idx = get_block_index(image.shape, idx, block_size)
            out_image[block_idx] = adaptive_median_threshold(image[block_idx])
    return out_image

# This function invokes the whole pipeline of Step 2.
def process_image(image_in):
    image_in = preprocess(image_in)
    image_out = block_image_process(image_in, BLOCK_SIZE)
    image_out = postprocess(image_out)
    return image_out
    
#############step 3######################
# This is the function used for composing
def sigmoid(x, orig, rad):
    k = np.exp((x - orig) * 2 / rad)
    return k / (k + 1.)

# Here, we combine the local blocks. A bit lengthy, so please
# follow the local comments.
def combine_block(img_in, mask):
    # First, we pre-fill the masked region of img_out to white
    # (i.e. background). The mask is retrieved from previous section.
    img_out = np.zeros_like(img_in)
    img_out[mask == 255] = 255
    fimg_in = img_in.astype(np.float32)

    # Then, we store the foreground (letters written with ink)
    # in the `idx` array. If there are none (i.e. just background),
    # we move on to the next block.
    idx = np.where(mask == 0)
    if idx[0].shape[0] == 0:
        img_out[idx] = img_in[idx]
        return img_out

    # We find the intensity range of our pixels in this local part
    # and clip the image block to that range, locally.
    lo = fimg_in[idx].min()
    hi = fimg_in[idx].max()
    v = fimg_in[idx] - lo
    r = hi - lo

    # Now we use good old OTSU binarization to get a rough estimation
    # of foreground and background regions.
    img_in_idx = img_in[idx]
    ret3,th3 = cv2.threshold(img_in[idx],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Then we normalize the stuffs and apply sigmoid to gradually
    # combine the stuffs.
    bound_value = np.min(img_in_idx[th3[:, 0] == 255])
    bound_value = (bound_value - lo) / (r + 1e-5)
    f = (v / (r + 1e-5))
    f = sigmoid(f, bound_value + 0.05, 0.2)

    # Finally, we re-normalize the result to the range [0..255]
    img_out[idx] = (255. * f).astype(np.uint8)
    return img_out

# We do the combination routine on local blocks, so that the scaling
# parameters of Sigmoid function can be adjusted to local setting
def combine_block_image_process(image, mask, block_size):
    out_image = np.zeros_like(image)
    for row in range(0, image.shape[0], block_size):
        for col in range(0, image.shape[1], block_size):
            idx = (row, col)
            block_idx = get_block_index(image.shape, idx, block_size)
            out_image[block_idx] = combine_block(
                image[block_idx], mask[block_idx])
    return out_image

# Postprocessing (should be robust even without it, but I recommend
# you to play around a bit and find what works best for your data.
# I just left it blank.
def combine_postprocess(image):
    return image

# The main function of this section. Executes the whole pipeline.
def combine_process(image_in, mask):
    image_out = combine_block_image_process(image_in, mask, 20)
    image_out = combine_postprocess(image_out)
    return image_out
#######################################
    
def load(data):
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray
        
def show(img):
    img = resize(img, 800)
    cv2.imshow('thresh', img)
    cv2.waitKey(0)
    
def resize(img, size):
    (h, w) = img.shape[:2]
    ratio = min(size / w, size / h)
    img = cv2.resize(img, (int(w * ratio), int(h * ratio)))
    return img
    
def portrait(img, _180):
    if img.shape[0] < img.shape[1]:
        img = np.rot90(img, 3)
        
    if _180:
        img = np.rot90(img, 2)
    return img
    
def numerics(text):
    parts = text.split()
    nums = [(re.match(r'^(\d*)([\.,-]\d\d)?([₪$]?)$', part), part) for part in parts]
    
    left = []
    results = []
    prev_result = None
    for i, (num, part) in enumerate(nums):
        if not num:
            left.append(part)
            continue
        n, frac, units = num.groups()
        if not n and not frac:
            #floating unit?
            left.append(part)
            continue
            
        is_money = bool(frac) or bool(units) or (not frac and len(n) == 2)
        
        result = None
        
        if n and frac:
            result = f'{n}.{frac[1:]}', True
        elif n and not frac and (bool(units) or len(n) == 2):
            #steal prev n if has prev and its not money
            if prev_result and not prev_result[1]:
                result = f'{prev_result[0]}.{n}', True
                prev_result = None
        elif not n and frac:
            #steal prev n if has prev and its not money
            if prev_result and not prev_result[1]:
                result = f'{prev_result[0]}.{frac[1:]}', True
                prev_result = None
        elif n and len(n) == 1: #only single digit for non-prices
            result = n, False
            
        if prev_result:
            results.append(prev_result)
        prev_result = result
    
    if prev_result:
        results.append(prev_result)
        
    return results, left
    
    
def process_text(lines):
    countreg = r'(\d+)'
    #TODO maybe without dot?
    
    pricereg = r'(\d{1,4}[ \.,-]\d{1,2})\s*[₪$]?'
    
    totalreg = r'(total|סך.*הכל|סה.{0,2}כ)'
    
    #TODO location based detection? by block_num (line[1])
    
    processed = []
    current = []
    for line in lines:
        text = line['text'].strip()
        if not text:
            continue
            
        print(text)
        
        '''
        match_start = re.match(f'^{pricereg}\s.*$', text)
        match_end = re.match(f'^.*\s{pricereg}$', text)
        if (not match_start and not match_end) or \
            (match_start and match_end): #wat?
            if current:
                processed.append(current)
                current = []
            continue
            
        if re.match(totalreg, text, re.I):
            #reached total, end this
            print('found total')
            break
            
        price_match = match_start or match_end
        count_match = re.match(f'^{countreg}.*$' if match_start is None else f'^.*{countreg}$', text)
        
        price = float(price_match.group(1).replace(',', '.').replace('-', '.'))
        count = int(count_match.group(1)) if count_match else 1
        if count > 9:
            count = 9 #sanity
            
        left = text.replace(price_match.group(1), '')
        left = left.replace(count_match.group(1), '') if count_match else left
        '''
        
        numeric_results, left = numerics(text)
        print(numeric_results)
        print(left)
        
        prices = [float(n) for n, is_money in numeric_results if is_money]
        counts = [int(n) for n, is_money in numeric_results if not is_money]
        
        if not prices:
            if current:
                processed.append(current)
                current = []
            continue
        
        if re.match(totalreg, text, re.I):
            #reached total, end this
            print('found total')
            break
            
        if not counts:
            count = 1
        elif len(counts) > 1:
            #what do?
            count = counts[0]
        else:
            count = counts[0]
            
        if len(prices) == 1:
            price = prices[0]
        elif len(prices) == 2 and count > 1:
            #probably total?
            price = min(prices)
        else:
            #what do?
            price = prices[-1]

        result = dict(price=price, count=count, entry=left, **line)
        current.append(result)
        
    if current:
        processed.append(current)
        
    #print(processed)
        
    if not processed:
        return None
        
    #largest block in processed is probably our list
    longest = max(processed, key=len)
    block_nums = set(entry['line'][1] for entry in longest)
    #add all in same blocks as longest
    results = [entry for block in processed for entry in block if entry['line'][1] in block_nums]
    
    '''
    longest_i, longest = max(enumerate(processed), key=lambda e: len(e[1]))
    #also add probable entries
    prepend = []
    append = []
    for i, block in enumerate(processed):
        lst = prepend if i < longest_i else (append if i > longest_i else []) #throwaway
        [lst.append(entry) for entry in block if entry['price'] > 5 and entry['count'] < 10]  
        
    results = [e for e in prepend + longest + append]
    '''
    
    return results
    
def to_alpha_png(img):
    #all black, invert alpha
    arr = np.ndarray((img.shape[0], img.shape[1], 4))
    arr[:,:,0] = np.zeros_like(img)
    arr[:,:,1] = np.zeros_like(img)
    arr[:,:,2] = np.zeros_like(img)
    arr[:,:,3] = 255 - img
    _, im = cv2.imencode('.PNG', arr)
    return im
    
def compress_spaces(img, min_space=50, max_space=100):
    pos = np.any(img != 255, axis=0)
    pos[0] = True
    pos[-1] = True
    diff = np.diff(pos.astype("int8"))

    starts, = np.where(diff < 0)
    ends, = np.where(diff > 0)
    sizes = ends - starts

    for start, size in zip(starts, sizes):
        if size >= max_space:
            pos[start:start + min_space] = True
        else:
            pos[start:start + size] = True
            
    return img.compress(pos, axis=1)
    
def pad(l, to, p):
    return l + (((to - len(l)) * [p]) if to > len(l) else [])
    
def extract_data(data, img):
    data_lines = data.splitlines()
    keys = data_lines[0].split('\t')
    rows = [dict(zip(keys, pad(data_line.split('\t'), len(keys), ''))) for data_line in data_lines[1:]]
    rows = [{k:(v if k == 'text' else int(v)) for k,v in row.items()} for row in rows]
    text_lines = sorted(list((key, list(group)) for key, group in itertools.groupby(rows, lambda e: (e['page_num'], e['block_num'], e['par_num'], e['line_num']))), key=lambda e: e[0])
    
    result_lines = []
    for key, text_line in text_lines:
        text_line.sort(key=lambda e: e['word_num'])
        left, top, width, height = (min(text_line, key=lambda e: e['left'])['left'], 
                                              min(text_line, key=lambda e: e['top'])['top'], 
                                              max(text_line, key=lambda e: e['width'])['width'], 
                                              max(text_line, key=lambda e: e['height'])['height'])
        
        confidence = sum(e['conf'] for e in text_line) // len(text_line)
        text = '  '.join(e['text'] for e in text_line).strip()
        image = to_alpha_png(compress_spaces(img[top : top + height, left : left + width])) if text else None
        result_lines.append(dict(left=left, top=top, width=width, height=height, line=key, confidence=confidence,
                                              text=text,
                                              image=image))
    
    return result_lines
    
def process_receipt(data, _180=False):
    orig_img = load(data)
    orig_img = resize(orig_img, 2000)
    orig_img = portrait(orig_img, _180)
    
    orig_img = adjust_gamma(orig_img)
    mask = process_image(orig_img)   
    img = combine_process(orig_img, mask)
    
    data = pytesseract.image_to_data(img, lang='heb+eng')
    lines = extract_data(data, img)
    return process_text(lines)
    
def main(argv):
    if len(argv) != 2:
        print(f'{argv[0]} <image in>')
        return
    
    data = open(argv[1], 'rb').read()
    process_receipt(data)
    
if __name__ == '__main__':
    main(sys.argv)
