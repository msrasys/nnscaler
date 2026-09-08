#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import codecs

# ast.unparse() lazily loads this codec while tracing. Load it before the tracer
# patches builtins.tuple, or construction of the tuple subclass CodecInfo fails.
codecs.lookup('unicode_escape')
