# Imports
# Logging
#
# Flow:
#     1. Read input JSON
#     2. Instantiate feature object
#     3. Apply get_feature() to each image
#     4. Write each feature to CSV
#
# Other:
#     * Image/open mode on input JSON
#     * Convert all images to same format so can use one image package
#
# Feature object:
#     * Get feature method
